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
def _calculate_aggregation_loss(logits_aggregation, aggregate_mask, aggregation_labels, use_answer_as_supervision, num_aggregation_labels, aggregation_loss_weight):
    """
    Calculates the aggregation loss per example.

    Args:
        logits_aggregation (`torch.FloatTensor` of shape `(batch_size, num_aggregation_labels)`):
            Logits per aggregation operation.
        aggregate_mask (`torch.FloatTensor` of shape `(batch_size, )`):
            A mask set to 1 for examples that should use aggregation functions.
        aggregation_labels (`torch.LongTensor` of shape `(batch_size, )`):
            Aggregation function id for every example in the batch.
        use_answer_as_supervision (`bool`, *optional*):
            Whether to use the answer as the only supervision for aggregation examples.
        num_aggregation_labels (`int`, *optional*, defaults to 0):
            The number of aggregation operators to predict.
        aggregation_loss_weight (`float`, *optional*, defaults to 1.0):
            Importance weight for the aggregation loss.

    Returns:
        aggregation_loss (`torch.FloatTensor` of shape `(batch_size,)`): Aggregation loss per example.
    """
    per_example_aggregation_loss = _calculate_aggregation_loss_known(logits_aggregation, aggregate_mask, aggregation_labels, use_answer_as_supervision, num_aggregation_labels)
    if use_answer_as_supervision:
        per_example_aggregation_loss += _calculate_aggregation_loss_unknown(logits_aggregation, aggregate_mask)
    return aggregation_loss_weight * per_example_aggregation_loss