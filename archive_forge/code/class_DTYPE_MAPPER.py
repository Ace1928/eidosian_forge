import functools
import random
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union
import numpy as np
from transformers.utils import is_tf_available, is_torch_available
from .normalized_config import (
class DTYPE_MAPPER:

    @classmethod
    def np(cls, dtype):
        mapping = {'fp32': np.float32, 'fp16': np.float16, 'int64': np.int64, 'int32': np.int32, 'int8': np.int8, 'bool': bool}
        return mapping[dtype]

    @classmethod
    def pt(cls, dtype):
        mapping = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16, 'int64': torch.int64, 'int32': torch.int32, 'int8': torch.int8, 'bool': torch.bool}
        return mapping[dtype]

    @classmethod
    def tf(cls, dtype):
        mapping = {'fp32': tf.float32, 'fp16': tf.float16, 'bf16': tf.bfloat16, 'int64': tf.int64, 'int32': tf.int32, 'int8': tf.int8, 'bool': tf.bool}
        return mapping[dtype]