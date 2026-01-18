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
@dataclass
class RewardDataCollatorWithPadding:
    """
    Reward DataCollator class that pads the inputs to the maximum length of the batch.
    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer used for encoding the data.
        padding (`Union[bool, str, `PaddingStrategy`]`, `optional`, defaults to `True`):
            padding_strategy to pass to the tokenizer.
        max_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the sequence to be processed.
        pad_to_multiple_of (`Optional[int]`, `optional`, defaults to `None`):
            If set will pad the sequence to a multiple of the provided value.
        return_tensors (`str`, `optional`, defaults to `"pt"`):
            The tensor type to use.
    """
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = 'pt'

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features_chosen = []
        features_rejected = []
        margin = []
        has_margin = 'margin' in features[0]
        for feature in features:
            if 'input_ids_chosen' not in feature or 'input_ids_rejected' not in feature or 'attention_mask_chosen' not in feature or ('attention_mask_rejected' not in feature):
                raise ValueError('The features should include `input_ids_chosen`, `attention_mask_chosen`, `input_ids_rejected` and `attention_mask_rejected`')
            features_chosen.append({'input_ids': feature['input_ids_chosen'], 'attention_mask': feature['attention_mask_chosen']})
            features_rejected.append({'input_ids': feature['input_ids_rejected'], 'attention_mask': feature['attention_mask_rejected']})
            if has_margin:
                margin.append(feature['margin'])
        batch_chosen = self.tokenizer.pad(features_chosen, padding=self.padding, max_length=self.max_length, pad_to_multiple_of=self.pad_to_multiple_of, return_tensors=self.return_tensors)
        batch_rejected = self.tokenizer.pad(features_rejected, padding=self.padding, max_length=self.max_length, pad_to_multiple_of=self.pad_to_multiple_of, return_tensors=self.return_tensors)
        batch = {'input_ids_chosen': batch_chosen['input_ids'], 'attention_mask_chosen': batch_chosen['attention_mask'], 'input_ids_rejected': batch_rejected['input_ids'], 'attention_mask_rejected': batch_rejected['attention_mask'], 'return_loss': True}
        if has_margin:
            margin = torch.tensor(margin, dtype=torch.float)
            batch['margin'] = margin
        return batch