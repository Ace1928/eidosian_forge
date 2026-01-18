import math
from typing import cast, Iterator, List, Optional, Sized, Union
import torch
import torch.distributed as dist
from torch.utils.data import Sampler
from torchvision.datasets.video_utils import VideoClips

    Samples at most `max_video_clips_per_video` clips for each video randomly

    Args:
        video_clips (VideoClips): video clips to sample from
        max_clips_per_video (int): maximum number of clips to be sampled per video
    