import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import torch
import transformers
import utils
from torch.utils.data import Dataset
from transformers import Trainer
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default='adamw_torch')
    model_max_length: int = field(default=512, metadata={'help': 'Maximum sequence length. Sequences will be right padded (and possibly truncated).'})