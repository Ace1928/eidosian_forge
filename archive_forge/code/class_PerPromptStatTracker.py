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
class PerPromptStatTracker:
    """
    Class for tracking statistics per prompt. Mainly used to calculate advantage for the DPPO algorithm

    Args:
        buffer_size (`int`):
            Size of the buffer to keep for each prompt.
        min_count (`int`):
            Minimum number of samples to keep in the buffer before calculating the mean and std.
    """

    def __init__(self, buffer_size, min_count):
        self.buffer_size = buffer_size
        self.min_count = min_count
        self.stats = {}

    def update(self, prompts, rewards):
        prompts = np.array(prompts)
        rewards = np.array(rewards)
        unique = np.unique(prompts)
        advantages = np.empty_like(rewards)
        for prompt in unique:
            prompt_rewards = rewards[prompts == prompt]
            if prompt not in self.stats:
                self.stats[prompt] = deque(maxlen=self.buffer_size)
            self.stats[prompt].extend(prompt_rewards)
            if len(self.stats[prompt]) < self.min_count:
                mean = np.mean(rewards)
                std = np.std(rewards) + 1e-06
            else:
                mean = np.mean(self.stats[prompt])
                std = np.std(self.stats[prompt]) + 1e-06
            advantages[prompts == prompt] = (prompt_rewards - mean) / std
        return advantages

    def get_stats(self):
        return {k: {'mean': np.mean(v), 'std': np.std(v), 'count': len(v)} for k, v in self.stats.items()}