import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch import Tensor
from torchaudio._internal import load_state_dict_from_url
from torchaudio.functional import mu_law_decoding
from torchaudio.models import Tacotron2, WaveRNN
from torchaudio.transforms import GriffinLim, InverseMelScale
from . import utils
from .interface import Tacotron2TTSBundle
class _PhoneMixin:

    def get_text_processor(self, *, dl_kwargs=None) -> Tacotron2TTSBundle.TextProcessor:
        return _EnglishPhoneProcessor(dl_kwargs=dl_kwargs)