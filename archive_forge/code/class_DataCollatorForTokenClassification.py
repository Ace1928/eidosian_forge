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
class DataCollatorForTokenClassification(DataCollatorMixin):
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = 'pt'

    def torch_call(self, features):
        import torch
        label_name = 'label' if 'label' in features[0].keys() else 'labels'
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
        no_labels_features = [{k: v for k, v in feature.items() if k != label_name} for feature in features]
        batch = pad_without_fast_tokenizer_warning(self.tokenizer, no_labels_features, padding=self.padding, max_length=self.max_length, pad_to_multiple_of=self.pad_to_multiple_of, return_tensors='pt')
        if labels is None:
            return batch
        sequence_length = batch['input_ids'].shape[1]
        padding_side = self.tokenizer.padding_side

        def to_list(tensor_or_iterable):
            if isinstance(tensor_or_iterable, torch.Tensor):
                return tensor_or_iterable.tolist()
            return list(tensor_or_iterable)
        if padding_side == 'right':
            batch[label_name] = [to_list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels]
        else:
            batch[label_name] = [[self.label_pad_token_id] * (sequence_length - len(label)) + to_list(label) for label in labels]
        batch[label_name] = torch.tensor(batch[label_name], dtype=torch.int64)
        return batch

    def tf_call(self, features):
        import tensorflow as tf
        label_name = 'label' if 'label' in features[0].keys() else 'labels'
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
        batch = pad_without_fast_tokenizer_warning(self.tokenizer, features, padding=self.padding, max_length=self.max_length, pad_to_multiple_of=self.pad_to_multiple_of, return_tensors='tf' if labels is None else None)
        if labels is None:
            return batch
        sequence_length = tf.convert_to_tensor(batch['input_ids']).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == 'right':
            batch['labels'] = [list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels]
        else:
            batch['labels'] = [[self.label_pad_token_id] * (sequence_length - len(label)) + list(label) for label in labels]
        batch = {k: tf.convert_to_tensor(v, dtype=tf.int64) for k, v in batch.items()}
        return batch

    def numpy_call(self, features):
        label_name = 'label' if 'label' in features[0].keys() else 'labels'
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
        batch = pad_without_fast_tokenizer_warning(self.tokenizer, features, padding=self.padding, max_length=self.max_length, pad_to_multiple_of=self.pad_to_multiple_of, return_tensors='np' if labels is None else None)
        if labels is None:
            return batch
        sequence_length = np.array(batch['input_ids']).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == 'right':
            batch['labels'] = [list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels]
        else:
            batch['labels'] = [[self.label_pad_token_id] * (sequence_length - len(label)) + list(label) for label in labels]
        batch = {k: np.array(v, dtype=np.int64) for k, v in batch.items()}
        return batch