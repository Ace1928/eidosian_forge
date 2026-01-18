from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, Iterator, Optional, Tuple, TypeVar, Union
import torch
import torio
from torch.utils._pytree import tree_map
@_format_audio_args
def add_basic_audio_stream(self, frames_per_chunk: int, buffer_chunk_size: int=3, *, stream_index: Optional[int]=None, decoder: Optional[str]=None, decoder_option: Optional[Dict[str, str]]=None, format: Optional[str]='fltp', sample_rate: Optional[int]=None, num_channels: Optional[int]=None):
    """Add output audio stream

        Args:
            frames_per_chunk (int): {frames_per_chunk}

            buffer_chunk_size (int, optional): {buffer_chunk_size}

            stream_index (int or None, optional): {stream_index}

            decoder (str or None, optional): {decoder}

            decoder_option (dict or None, optional): {decoder_option}

            format (str, optional): Output sample format (precision).

                If ``None``, the output chunk has dtype corresponding to
                the precision of the source audio.

                Otherwise, the sample is converted and the output dtype is changed
                as following.

                - ``"u8p"``: The output is ``torch.uint8`` type.
                - ``"s16p"``: The output is ``torch.int16`` type.
                - ``"s32p"``: The output is ``torch.int32`` type.
                - ``"s64p"``: The output is ``torch.int64`` type.
                - ``"fltp"``: The output is ``torch.float32`` type.
                - ``"dblp"``: The output is ``torch.float64`` type.

                Default: ``"fltp"``.

            sample_rate (int or None, optional): If provided, resample the audio.

            num_channels (int, or None, optional): If provided, change the number of channels.
        """
    self.add_audio_stream(frames_per_chunk, buffer_chunk_size, stream_index=stream_index, decoder=decoder, decoder_option=decoder_option, filter_desc=_get_afilter_desc(sample_rate, format, num_channels))