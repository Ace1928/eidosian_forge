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
class ModelArguments:
    model_name_or_path: Optional[str] = field(default='facebook/opt-125m')