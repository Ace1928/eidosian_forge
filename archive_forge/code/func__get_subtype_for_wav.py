import warnings
from typing import Optional, Tuple
import torch
from torchaudio._internal import module_utils as _mod_utils
from .common import AudioMetaData
def _get_subtype_for_wav(dtype: torch.dtype, encoding: str, bits_per_sample: int):
    if not encoding:
        if not bits_per_sample:
            subtype = {torch.uint8: 'PCM_U8', torch.int16: 'PCM_16', torch.int32: 'PCM_32', torch.float32: 'FLOAT', torch.float64: 'DOUBLE'}.get(dtype)
            if not subtype:
                raise ValueError(f'Unsupported dtype for wav: {dtype}')
            return subtype
        if bits_per_sample == 8:
            return 'PCM_U8'
        return f'PCM_{bits_per_sample}'
    if encoding == 'PCM_S':
        if not bits_per_sample:
            return 'PCM_32'
        if bits_per_sample == 8:
            raise ValueError('wav does not support 8-bit signed PCM encoding.')
        return f'PCM_{bits_per_sample}'
    if encoding == 'PCM_U':
        if bits_per_sample in (None, 8):
            return 'PCM_U8'
        raise ValueError('wav only supports 8-bit unsigned PCM encoding.')
    if encoding == 'PCM_F':
        if bits_per_sample in (None, 32):
            return 'FLOAT'
        if bits_per_sample == 64:
            return 'DOUBLE'
        raise ValueError('wav only supports 32/64-bit float PCM encoding.')
    if encoding == 'ULAW':
        if bits_per_sample in (None, 8):
            return 'ULAW'
        raise ValueError('wav only supports 8-bit mu-law encoding.')
    if encoding == 'ALAW':
        if bits_per_sample in (None, 8):
            return 'ALAW'
        raise ValueError('wav only supports 8-bit a-law encoding.')
    raise ValueError(f'wav does not support {encoding}.')