import math
import warnings
from fractions import Fraction
from typing import Dict, List, Optional, Tuple, Union
import torch
from ..extension import _load_library
def _read_video_from_memory(video_data: torch.Tensor, seek_frame_margin: float=0.25, read_video_stream: int=1, video_width: int=0, video_height: int=0, video_min_dimension: int=0, video_max_dimension: int=0, video_pts_range: Tuple[int, int]=(0, -1), video_timebase_numerator: int=0, video_timebase_denominator: int=1, read_audio_stream: int=1, audio_samples: int=0, audio_channels: int=0, audio_pts_range: Tuple[int, int]=(0, -1), audio_timebase_numerator: int=0, audio_timebase_denominator: int=1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reads a video from memory, returning both the video frames as the audio frames
    This function is torchscriptable.

    Args:
    video_data (data type could be 1) torch.Tensor, dtype=torch.int8 or 2) python bytes):
        compressed video content stored in either 1) torch.Tensor 2) python bytes
    seek_frame_margin (double, optional): seeking frame in the stream is imprecise.
        Thus, when video_start_pts is specified, we seek the pts earlier by seek_frame_margin seconds
    read_video_stream (int, optional): whether read video stream. If yes, set to 1. Otherwise, 0
    video_width/video_height/video_min_dimension/video_max_dimension (int): together decide
        the size of decoded frames:

            - When video_width = 0, video_height = 0, video_min_dimension = 0,
                and video_max_dimension = 0, keep the original frame resolution
            - When video_width = 0, video_height = 0, video_min_dimension != 0,
                and video_max_dimension = 0, keep the aspect ratio and resize the
                frame so that shorter edge size is video_min_dimension
            - When video_width = 0, video_height = 0, video_min_dimension = 0,
                and video_max_dimension != 0, keep the aspect ratio and resize
                the frame so that longer edge size is video_max_dimension
            - When video_width = 0, video_height = 0, video_min_dimension != 0,
                and video_max_dimension != 0, resize the frame so that shorter
                edge size is video_min_dimension, and longer edge size is
                video_max_dimension. The aspect ratio may not be preserved
            - When video_width = 0, video_height != 0, video_min_dimension = 0,
                and video_max_dimension = 0, keep the aspect ratio and resize
                the frame so that frame video_height is $video_height
            - When video_width != 0, video_height == 0, video_min_dimension = 0,
                and video_max_dimension = 0, keep the aspect ratio and resize
                the frame so that frame video_width is $video_width
            - When video_width != 0, video_height != 0, video_min_dimension = 0,
                and video_max_dimension = 0, resize the frame so that frame
                video_width and  video_height are set to $video_width and
                $video_height, respectively
    video_pts_range (list(int), optional): the start and end presentation timestamp of video stream
    video_timebase_numerator / video_timebase_denominator (float, optional): a rational
        number which denotes timebase in video stream
    read_audio_stream (int, optional): whether read audio stream. If yes, set to 1. Otherwise, 0
    audio_samples (int, optional): audio sampling rate
    audio_channels (int optional): audio audio_channels
    audio_pts_range (list(int), optional): the start and end presentation timestamp of audio stream
    audio_timebase_numerator / audio_timebase_denominator (float, optional):
        a rational number which denotes time base in audio stream

    Returns:
        vframes (Tensor[T, H, W, C]): the `T` video frames
        aframes (Tensor[L, K]): the audio frames, where `L` is the number of points and
            `K` is the number of channels
    """
    _validate_pts(video_pts_range)
    _validate_pts(audio_pts_range)
    if not isinstance(video_data, torch.Tensor):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='The given buffer is not writable')
            video_data = torch.frombuffer(video_data, dtype=torch.uint8)
    result = torch.ops.video_reader.read_video_from_memory(video_data, seek_frame_margin, 0, read_video_stream, video_width, video_height, video_min_dimension, video_max_dimension, video_pts_range[0], video_pts_range[1], video_timebase_numerator, video_timebase_denominator, read_audio_stream, audio_samples, audio_channels, audio_pts_range[0], audio_pts_range[1], audio_timebase_numerator, audio_timebase_denominator)
    vframes, _vframe_pts, vtimebase, vfps, vduration, aframes, aframe_pts, atimebase, asample_rate, aduration = result
    if aframes.numel() > 0:
        aframes = _align_audio_frames(aframes, aframe_pts, audio_pts_range)
    return (vframes, aframes)