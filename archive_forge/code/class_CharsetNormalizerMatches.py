import warnings
from typing import Dict, Optional, Union
from .api import from_bytes, from_fp, from_path, normalize
from .constant import CHARDET_CORRESPONDENCE
from .models import CharsetMatch, CharsetMatches
class CharsetNormalizerMatches(CharsetMatches):

    @staticmethod
    def from_fp(*args, **kwargs):
        warnings.warn('staticmethod from_fp, from_bytes, from_path and normalize are deprecated and scheduled to be removed in 3.0', DeprecationWarning)
        return from_fp(*args, **kwargs)

    @staticmethod
    def from_bytes(*args, **kwargs):
        warnings.warn('staticmethod from_fp, from_bytes, from_path and normalize are deprecated and scheduled to be removed in 3.0', DeprecationWarning)
        return from_bytes(*args, **kwargs)

    @staticmethod
    def from_path(*args, **kwargs):
        warnings.warn('staticmethod from_fp, from_bytes, from_path and normalize are deprecated and scheduled to be removed in 3.0', DeprecationWarning)
        return from_path(*args, **kwargs)

    @staticmethod
    def normalize(*args, **kwargs):
        warnings.warn('staticmethod from_fp, from_bytes, from_path and normalize are deprecated and scheduled to be removed in 3.0', DeprecationWarning)
        return normalize(*args, **kwargs)