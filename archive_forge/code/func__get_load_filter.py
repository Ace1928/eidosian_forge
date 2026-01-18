import os
import re
import sys
from typing import BinaryIO, Optional, Tuple, Union
import torch
import torchaudio
from .backend import Backend
from .common import AudioMetaData
def _get_load_filter(frame_offset: int=0, num_frames: int=-1, convert: bool=True) -> Optional[str]:
    if frame_offset < 0:
        raise RuntimeError('Invalid argument: frame_offset must be non-negative. Found: {}'.format(frame_offset))
    if num_frames == 0 or num_frames < -1:
        raise RuntimeError('Invalid argument: num_frames must be -1 or greater than 0. Found: {}'.format(num_frames))
    if frame_offset == 0 and num_frames == -1 and (not convert):
        return None
    aformat = 'aformat=sample_fmts=fltp'
    if frame_offset == 0 and num_frames == -1 and convert:
        return aformat
    if num_frames > 0:
        atrim = 'atrim=start_sample={}:end_sample={}'.format(frame_offset, frame_offset + num_frames)
    else:
        atrim = 'atrim=start_sample={}'.format(frame_offset)
    if not convert:
        return atrim
    return '{},{}'.format(atrim, aformat)