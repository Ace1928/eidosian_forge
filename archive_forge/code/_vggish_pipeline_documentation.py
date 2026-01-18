from dataclasses import dataclass
from typing import Callable, Dict
import torch
import torchaudio
from ._vggish_impl import _SAMPLE_RATE, VGGish as _VGGish, VGGishInputProcessor as _VGGishInputProcessor
Constructs input processor for VGGish.

        Returns:
            VGGishInputProcessor: input processor for VGGish.
        