import random
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from random import randint
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import numpy as np
from ..models.bert import BertTokenizer, BertTokenizerFast
from ..tokenization_utils_base import PreTrainedTokenizerBase
from ..utils import PaddingStrategy
@dataclass
class DataCollatorForSOP(DataCollatorForLanguageModeling):
    """
    Data collator used for sentence order prediction task.

    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for both masked language modeling and sentence order prediction
    """

    def __init__(self, *args, **kwargs):
        warnings.warn('DataCollatorForSOP is deprecated and will be removed in a future version, you can now use DataCollatorForLanguageModeling instead.', FutureWarning)

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        import torch
        from torch.nn.utils.rnn import pad_sequence
        input_ids = [example['input_ids'] for example in examples]
        input_ids = _torch_collate_batch(input_ids, self.tokenizer)
        input_ids, labels, attention_mask = self.mask_tokens(input_ids)
        token_type_ids = [example['token_type_ids'] for example in examples]
        token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        sop_label_list = [example['sentence_order_label'] for example in examples]
        sentence_order_label = torch.stack(sop_label_list)
        return {'input_ids': input_ids, 'labels': labels, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids, 'sentence_order_label': sentence_order_label}

    def mask_tokens(self, inputs: Any) -> Tuple[Any, Any, Any]:
        """
        Prepare masked tokens inputs/labels/attention_mask for masked language modeling: 80% MASK, 10% random, 10%
        original. N-gram not applied yet.
        """
        import torch
        if self.tokenizer.mask_token is None:
            raise ValueError('This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer.')
        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        attention_mask = (~masked_indices).float()
        if self.tokenizer._pad_token is not None:
            attention_padding_mask = labels.eq(self.tokenizer.pad_token_id)
            attention_mask.masked_fill_(attention_padding_mask, value=1.0)
        labels[~masked_indices] = -100
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        return (inputs, labels, attention_mask)