from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, Optional, Union
import torch
import torio
@dataclass
class CodecConfig:
    """Codec configuration."""
    bit_rate: int = -1
    'Bit rate'
    compression_level: int = -1
    'Compression level'
    qscale: Optional[int] = None
    'Global quality factor. Enables variable bit rate. Valid values depend on encoder.\n\n    For example: MP3 takes ``0`` - ``9`` (https://trac.ffmpeg.org/wiki/Encode/MP3) while\n    libvorbis takes ``-1`` - ``10``.\n    '
    gop_size: int = -1
    'The number of pictures in a group of pictures, or 0 for intra_only'
    max_b_frames: int = -1
    'maximum number of B-frames between non-B-frames.'