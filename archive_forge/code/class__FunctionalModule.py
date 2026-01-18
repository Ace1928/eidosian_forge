import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Callable, List, Tuple
import torch
import torchaudio
from torchaudio._internal import module_utils
from torchaudio.models import emformer_rnnt_base, RNNT, RNNTBeamSearch
class _FunctionalModule(torch.nn.Module):

    def __init__(self, functional):
        super().__init__()
        self.functional = functional

    def forward(self, input):
        return self.functional(input)