from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, Iterator, Optional, Tuple, TypeVar, Union
import torch
import torio
from torch.utils._pytree import tree_map
@_format_audio_args
def add_audio_stream(self, frames_per_chunk: int, buffer_chunk_size: int=3, *, stream_index: Optional[int]=None, decoder: Optional[str]=None, decoder_option: Optional[Dict[str, str]]=None, filter_desc: Optional[str]=None):
    """Add output audio stream

        Args:
            frames_per_chunk (int): {frames_per_chunk}

            buffer_chunk_size (int, optional): {buffer_chunk_size}

            stream_index (int or None, optional): {stream_index}

            decoder (str or None, optional): {decoder}

            decoder_option (dict or None, optional): {decoder_option}

            filter_desc (str or None, optional): Filter description.
                The list of available filters can be found at
                https://ffmpeg.org/ffmpeg-filters.html
                Note that complex filters are not supported.

        """
    i = self.default_audio_stream if stream_index is None else stream_index
    if i is None:
        raise RuntimeError('There is no audio stream.')
    self._be.add_audio_stream(i, frames_per_chunk, buffer_chunk_size, filter_desc, decoder, decoder_option or {})