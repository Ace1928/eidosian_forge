from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, Iterator, Optional, Tuple, TypeVar, Union
import torch
import torio
from torch.utils._pytree import tree_map
@dataclass
class OutputAudioStream(OutputStream):
    """Information about an audio output stream configured with
    :meth:`~torio.io.StreamingMediaDecoder.add_audio_stream` or
    :meth:`~torio.io.StreamingMediaDecoder.add_basic_audio_stream`.

    In addition to the attributes reported by :class:`OutputStream`,
    the following attributes are reported.
    """
    sample_rate: float
    'Sample rate of the audio.'
    num_channels: int
    'Number of channels.'