from __future__ import annotations
import itertools as it
from abc import abstractmethod
from collections import namedtuple
from typing import Dict, List, NamedTuple, Optional, Tuple, Union
import torch
from flashlight.lib.text.decoder import (
from flashlight.lib.text.dictionary import (
from torchaudio.utils import download_asset
def _get_timesteps(self, idxs: torch.IntTensor) -> torch.IntTensor:
    """Returns frame numbers corresponding to non-blank tokens."""
    timesteps = []
    for i, idx in enumerate(idxs):
        if idx == self.blank:
            continue
        if i == 0 or idx != idxs[i - 1]:
            timesteps.append(i)
    return torch.IntTensor(timesteps)