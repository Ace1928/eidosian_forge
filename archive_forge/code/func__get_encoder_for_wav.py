import os
import re
import sys
from typing import BinaryIO, Optional, Tuple, Union
import torch
import torchaudio
from .backend import Backend
from .common import AudioMetaData
def _get_encoder_for_wav(encoding: str, bits_per_sample: int) -> str:
    if bits_per_sample not in {None, 8, 16, 24, 32, 64}:
        raise ValueError(f'Invalid bits_per_sample {bits_per_sample} for WAV encoding.')
    endianness = _native_endianness()
    if not encoding:
        if not bits_per_sample:
            return f'pcm_s16{endianness}'
        if bits_per_sample == 8:
            return 'pcm_u8'
        return f'pcm_s{bits_per_sample}{endianness}'
    if encoding == 'PCM_S':
        if not bits_per_sample:
            bits_per_sample = 16
        if bits_per_sample == 8:
            raise ValueError('For WAV signed PCM, 8-bit encoding is not supported.')
        return f'pcm_s{bits_per_sample}{endianness}'
    if encoding == 'PCM_U':
        if bits_per_sample in (None, 8):
            return 'pcm_u8'
        raise ValueError('For WAV unsigned PCM, only 8-bit encoding is supported.')
    if encoding == 'PCM_F':
        if not bits_per_sample:
            bits_per_sample = 32
        if bits_per_sample in (32, 64):
            return f'pcm_f{bits_per_sample}{endianness}'
        raise ValueError('For WAV float PCM, only 32- and 64-bit encodings are supported.')
    if encoding == 'ULAW':
        if bits_per_sample in (None, 8):
            return 'pcm_mulaw'
        raise ValueError('For WAV PCM mu-law, only 8-bit encoding is supported.')
    if encoding == 'ALAW':
        if bits_per_sample in (None, 8):
            return 'pcm_alaw'
        raise ValueError('For WAV PCM A-law, only 8-bit encoding is supported.')
    raise ValueError(f'WAV encoding {encoding} is not supported.')