import warnings
from typing import Optional, Tuple
import torch
from torchaudio._internal import module_utils as _mod_utils
from .common import AudioMetaData
def _get_subtype_for_sphere(encoding: str, bits_per_sample: int):
    if encoding in (None, 'PCM_S'):
        return f'PCM_{bits_per_sample}' if bits_per_sample else 'PCM_32'
    if encoding in ('PCM_U', 'PCM_F'):
        raise ValueError(f'sph does not support {encoding} encoding.')
    if encoding == 'ULAW':
        if bits_per_sample in (None, 8):
            return 'ULAW'
        raise ValueError('sph only supports 8-bit for mu-law encoding.')
    if encoding == 'ALAW':
        return 'ALAW'
    raise ValueError(f'sph does not support {encoding}.')