from __future__ import annotations
import functools
import gc
import inspect
import json
import os
import pickle
import re
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union
import h5py
import numpy as np
import tensorflow as tf
from packaging.version import parse
from . import DataCollatorWithPadding, DefaultDataCollator
from .activations_tf import get_tf_activation
from .configuration_utils import PretrainedConfig
from .dynamic_module_utils import custom_object_save
from .generation import GenerationConfig, TFGenerationMixin
from .tf_utils import (
from .utils import (
from .utils.hub import convert_file_size_to_int, get_checkpoint_shard_files
class TFConv1D(keras.layers.Layer):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (`int`):
            The number of output features.
        nx (`int`):
            The number of input features.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation to use to initialize the weights.
        kwargs (`Dict[str, Any]`, *optional*):
            Additional keyword arguments passed along to the `__init__` of `keras.layers.Layer`.
    """

    def __init__(self, nf, nx, initializer_range=0.02, **kwargs):
        super().__init__(**kwargs)
        self.nf = nf
        self.nx = nx
        self.initializer_range = initializer_range

    def build(self, input_shape):
        if self.built:
            return
        self.built = True
        self.weight = self.add_weight('weight', shape=[self.nx, self.nf], initializer=get_initializer(self.initializer_range))
        self.bias = self.add_weight('bias', shape=[1, self.nf], initializer=tf.zeros_initializer())

    def call(self, x):
        bz, sl = shape_list(x)[:2]
        x = tf.reshape(x, [-1, self.nx])
        x = tf.matmul(x, self.weight) + self.bias
        x = tf.reshape(x, [bz, sl, self.nf])
        return x