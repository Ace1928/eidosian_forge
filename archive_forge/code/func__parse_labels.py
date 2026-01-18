import inspect
from typing import List, Union
import numpy as np
from ..tokenization_utils import TruncationStrategy
from ..utils import add_end_docstrings, logging
from .base import ArgumentHandler, ChunkPipeline, build_pipeline_init_args
def _parse_labels(self, labels):
    if isinstance(labels, str):
        labels = [label.strip() for label in labels.split(',') if label.strip()]
    return labels