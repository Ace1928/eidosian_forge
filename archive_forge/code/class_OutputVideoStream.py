from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, Iterator, Optional, Tuple, TypeVar, Union
import torch
import torio
from torch.utils._pytree import tree_map
@dataclass
class OutputVideoStream(OutputStream):
    """Information about a video output stream configured with
    :meth:`~torio.io.StreamingMediaDecoder.add_video_stream` or
    :meth:`~torio.io.StreamingMediaDecoder.add_basic_video_stream`.

    In addition to the attributes reported by :class:`OutputStream`,
    the following attributes are reported.
    """
    width: int
    'Width of the video frame in pixel.'
    height: int
    'Height of the video frame in pixel.'
    frame_rate: float
    'Frame rate.'