import enum
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, MaskedLMOutput, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import (
from ...utils import (
from .configuration_tapas import TapasConfig
class ProductIndexMap(IndexMap):
    """The product of two indices."""

    def __init__(self, outer_index, inner_index):
        """
        Combines indices i and j into pairs (i, j). The result is an index where each segment (i, j) is the
        intersection of segments i and j. For example if the inputs represent table cells indexed by respectively rows
        and columns the output will be a table indexed by (row, column) pairs, i.e. by cell. The implementation
        combines indices {0, .., n - 1} and {0, .., m - 1} into {0, .., nm - 1}. The output has *num_segments* equal to
        *outer_index.num_segments* * *inner_index.num_segments*

        Args:
            outer_index (`IndexMap`):
                IndexMap.
            inner_index (`IndexMap`):
                IndexMap, must have the same shape as *outer_index*.
        """
        if outer_index.batch_dims != inner_index.batch_dims:
            raise ValueError('outer_index.batch_dims and inner_index.batch_dims must be the same.')
        super().__init__(indices=inner_index.indices + outer_index.indices * inner_index.num_segments, num_segments=inner_index.num_segments * outer_index.num_segments, batch_dims=inner_index.batch_dims)
        self.outer_index = outer_index
        self.inner_index = inner_index

    def project_outer(self, index):
        """Projects an index with the same index set onto the outer components."""
        indices = torch.div(index.indices, self.inner_index.num_segments, rounding_mode='floor').type(torch.long)
        return IndexMap(indices=indices, num_segments=self.outer_index.num_segments, batch_dims=index.batch_dims)

    def project_inner(self, index):
        """Projects an index with the same index set onto the inner components."""
        return IndexMap(indices=torch.fmod(index.indices, self.inner_index.num_segments).type(torch.float).floor().type(torch.long), num_segments=self.inner_index.num_segments, batch_dims=index.batch_dims)