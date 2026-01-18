import math
from typing import cast, Iterator, List, Optional, Sized, Union
import torch
import torch.distributed as dist
from torch.utils.data import Sampler
from torchvision.datasets.video_utils import VideoClips
class RandomClipSampler(Sampler):
    """
    Samples at most `max_video_clips_per_video` clips for each video randomly

    Args:
        video_clips (VideoClips): video clips to sample from
        max_clips_per_video (int): maximum number of clips to be sampled per video
    """

    def __init__(self, video_clips: VideoClips, max_clips_per_video: int) -> None:
        if not isinstance(video_clips, VideoClips):
            raise TypeError(f'Expected video_clips to be an instance of VideoClips, got {type(video_clips)}')
        self.video_clips = video_clips
        self.max_clips_per_video = max_clips_per_video

    def __iter__(self) -> Iterator[int]:
        idxs = []
        s = 0
        for c in self.video_clips.clips:
            length = len(c)
            size = min(length, self.max_clips_per_video)
            sampled = torch.randperm(length)[:size] + s
            s += length
            idxs.append(sampled)
        idxs_ = torch.cat(idxs)
        perm = torch.randperm(len(idxs_))
        return iter(idxs_[perm].tolist())

    def __len__(self) -> int:
        return sum((min(len(c), self.max_clips_per_video) for c in self.video_clips.clips))