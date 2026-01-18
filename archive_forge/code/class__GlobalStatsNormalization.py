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
class _GlobalStatsNormalization(torch.nn.Module):

    def __init__(self, global_stats_path):
        super().__init__()
        with open(global_stats_path) as f:
            blob = json.loads(f.read())
        self.register_buffer('mean', torch.tensor(blob['mean']))
        self.register_buffer('invstddev', torch.tensor(blob['invstddev']))

    def forward(self, input):
        return (input - self.mean) * self.invstddev