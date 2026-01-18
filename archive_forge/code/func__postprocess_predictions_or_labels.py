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
def _postprocess_predictions_or_labels(self, inputs):
    if isinstance(inputs[0], dict):
        outputs = {}
        for key in inputs[0].keys():
            outputs[key] = self._concatenate_batches([batch[key] for batch in inputs])
        if len(outputs) == 1:
            outputs = list(outputs.values())[0]
    elif isinstance(inputs[0], list) or isinstance(inputs[0], tuple):
        outputs = []
        for input_list in zip(*inputs):
            outputs.append(self._concatenate_batches(input_list))
        if len(outputs) == 1:
            outputs = outputs[0]
    elif isinstance(inputs[0], np.ndarray):
        outputs = self._concatenate_batches(inputs)
    elif isinstance(inputs[0], tf.Tensor):
        outputs = self._concatenate_batches([tensor.numpy() for tensor in inputs])
    else:
        raise TypeError(f"Couldn't handle batch of type {type(inputs[0])}!")
    return outputs