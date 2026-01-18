import argparse
import random
from pathlib import Path
from typing import Optional, Union, Tuple, List, Any, Callable
import torch
import torch.utils.data
import torchaudio
import tqdm
def _augment_force_stereo(audio: torch.Tensor) -> torch.Tensor:
    if audio.shape[0] > 2:
        audio = audio[:2, ...]
    if audio.shape[0] == 1:
        audio = torch.repeat_interleave(audio, 2, dim=0)
    return audio