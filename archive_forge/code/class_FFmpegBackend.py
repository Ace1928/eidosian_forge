import os
import re
import sys
from typing import BinaryIO, Optional, Tuple, Union
import torch
import torchaudio
from .backend import Backend
from .common import AudioMetaData
class FFmpegBackend(Backend):

    @staticmethod
    def info(uri: InputType, format: Optional[str], buffer_size: int=4096) -> AudioMetaData:
        metadata = info_audio(uri, format, buffer_size)
        metadata.bits_per_sample = _get_bits_per_sample(metadata.encoding, metadata.bits_per_sample)
        metadata.encoding = _map_encoding(metadata.encoding)
        return metadata

    @staticmethod
    def load(uri: InputType, frame_offset: int=0, num_frames: int=-1, normalize: bool=True, channels_first: bool=True, format: Optional[str]=None, buffer_size: int=4096) -> Tuple[torch.Tensor, int]:
        return load_audio(uri, frame_offset, num_frames, normalize, channels_first, format)

    @staticmethod
    def save(uri: InputType, src: torch.Tensor, sample_rate: int, channels_first: bool=True, format: Optional[str]=None, encoding: Optional[str]=None, bits_per_sample: Optional[int]=None, buffer_size: int=4096, compression: Optional[Union[torchaudio.io.CodecConfig, float, int]]=None) -> None:
        if not isinstance(compression, (torchaudio.io.CodecConfig, type(None))):
            raise ValueError('FFmpeg backend expects non-`None` value for argument `compression` to be of ', f'type `torchaudio.io.CodecConfig`, but received value of type {type(compression)}')
        save_audio(uri, src, sample_rate, channels_first, format, encoding, bits_per_sample, buffer_size, compression)

    @staticmethod
    def can_decode(uri: InputType, format: Optional[str]) -> bool:
        return True

    @staticmethod
    def can_encode(uri: InputType, format: Optional[str]) -> bool:
        return True