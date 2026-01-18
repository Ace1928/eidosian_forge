import logging
import os
from pathlib import Path
from time import sleep
from typing import Callable, List, Optional, Union
import numpy as np
import tensorflow as tf
from huggingface_hub import Repository, create_repo
from packaging.version import parse
from . import IntervalStrategy, PreTrainedTokenizerBase
from .modelcard import TrainingSummary
from .modeling_tf_utils import keras
@staticmethod
def _concatenate_batches(batches, padding_index=-100):
    if batches[0].ndim == 1 or all((batch.shape[1] == batches[0].shape[1] for batch in batches)):
        return np.concatenate(batches, axis=0)
    max_len = max([batch.shape[1] for batch in batches])
    num_samples = sum([batch.shape[0] for batch in batches])
    output = np.full_like(batches[0], fill_value=padding_index, shape=[num_samples, max_len] + list(batches[0].shape[2:]))
    i = 0
    for batch in batches:
        output[i:i + len(batch), :batch.shape[1]] = batch
        i += len(batch)
    return output