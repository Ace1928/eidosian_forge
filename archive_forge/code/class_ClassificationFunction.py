import inspect
import warnings
from typing import Dict
import numpy as np
from ..utils import ExplicitEnum, add_end_docstrings, is_tf_available, is_torch_available
from .base import GenericTensor, Pipeline, build_pipeline_init_args
class ClassificationFunction(ExplicitEnum):
    SIGMOID = 'sigmoid'
    SOFTMAX = 'softmax'
    NONE = 'none'