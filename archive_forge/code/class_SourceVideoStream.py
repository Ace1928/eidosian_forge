from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, Iterator, Optional, Tuple, TypeVar, Union
import torch
import torio
from torch.utils._pytree import tree_map
@dataclass
class SourceVideoStream(SourceStream):
    """The metadata of a video source stream, returned by :meth:`~torio.io.StreamingMediaDecoder.get_src_stream_info`.

    This class is used when representing video stream.

    In addition to the attributes reported by :class:`SourceStream`,
    the following attributes are reported.
    """
    width: int
    'Width of the video frame in pixel.'
    height: int
    'Height of the video frame in pixel.'
    frame_rate: float
    'Frame rate.'