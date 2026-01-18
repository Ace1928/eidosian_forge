import random
import warnings
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from accelerate import PartialState
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset
from transformers import BitsAndBytesConfig, DataCollatorForLanguageModeling, PreTrainedTokenizerBase
from ..import_utils import is_peft_available, is_unsloth_available, is_xpu_available
from ..trainer.model_config import ModelConfig
class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target, horizon):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current, n_steps):
        target = self.target
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult