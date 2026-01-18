from dataclasses import dataclass
from typing import Callable, Dict
import torch
import torchaudio
from ._vggish_impl import _SAMPLE_RATE, VGGish as _VGGish, VGGishInputProcessor as _VGGishInputProcessor
class VGGish(_VGGish):
    __doc__ = _VGGish.__doc__