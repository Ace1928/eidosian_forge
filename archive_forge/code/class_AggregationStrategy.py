import types
import warnings
from typing import List, Optional, Tuple, Union
import numpy as np
from ..models.bert.tokenization_bert import BasicTokenizer
from ..utils import (
from .base import ArgumentHandler, ChunkPipeline, Dataset, build_pipeline_init_args
class AggregationStrategy(ExplicitEnum):
    """All the valid aggregation strategies for TokenClassificationPipeline"""
    NONE = 'none'
    SIMPLE = 'simple'
    FIRST = 'first'
    AVERAGE = 'average'
    MAX = 'max'