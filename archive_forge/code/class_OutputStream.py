from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, Iterator, Optional, Tuple, TypeVar, Union
import torch
import torio
from torch.utils._pytree import tree_map
@dataclass
class OutputStream:
    """Output stream configured on :class:`StreamingMediaDecoder`,
    returned by :meth:`~torio.io.StreamingMediaDecoder.get_out_stream_info`.
    """
    source_index: int
    'Index of the source stream that this output stream is connected.'
    filter_description: str
    'Description of filter graph applied to the source stream.'
    media_type: str
    'The type of the stream. ``"audio"`` or ``"video"``.'
    format: str
    'Media format. Such as ``"s16"`` and ``"yuv420p"``.\n\n    Commonly found audio values are;\n\n    - ``"u8"``, ``"u8p"``: Unsigned 8-bit unsigned interger.\n    - ``"s16"``, ``"s16p"``: 16-bit signed integer.\n    - ``"s32"``, ``"s32p"``: 32-bit signed integer.\n    - ``"flt"``, ``"fltp"``: 32-bit floating-point.\n\n    .. note::\n\n       `p` at the end indicates the format is `planar`.\n       Channels are grouped together instead of interspersed in memory.'