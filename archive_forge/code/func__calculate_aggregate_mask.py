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
def _calculate_aggregate_mask(answer, pooled_output, cell_selection_preference, labels, aggregation_classifier):
    """
    Finds examples where the model should select cells with no aggregation.

    Returns a mask that determines for which examples should the model select answers directly from the table, without
    any aggregation function. If the answer is a piece of text the case is unambiguous as aggregation functions only
    apply to numbers. If the answer is a number but does not appear in the table then we must use some aggregation
    case. The ambiguous case is when the answer is a number that also appears in the table. In this case we use the
    aggregation function probabilities predicted by the model to decide whether to select or aggregate. The threshold
    for this is a hyperparameter *cell_selection_preference*

    Args:
        answer (`torch.FloatTensor` of shape `(batch_size, )`):
            Answer for every example in the batch. Nan if there is no scalar answer.
        pooled_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            Output of the pooler (BertPooler) on top of the encoder layer.
        cell_selection_preference (`float`):
            Preference for cell selection in ambiguous cases.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Labels per token. aggregation_classifier (`torch.nn.Linear`): Aggregation head

    Returns:
        aggregate_mask (`torch.FloatTensor` of shape `(batch_size,)`): A mask set to 1 for examples that should use
        aggregation functions.
    """
    aggregate_mask_init = torch.logical_not(torch.isnan(answer)).type(torch.FloatTensor).to(answer.device)
    logits_aggregation = aggregation_classifier(pooled_output)
    dist_aggregation = torch.distributions.categorical.Categorical(logits=logits_aggregation)
    aggregation_ops_total_mass = torch.sum(dist_aggregation.probs[:, 1:], dim=1)
    is_pred_cell_selection = aggregation_ops_total_mass <= cell_selection_preference
    is_cell_supervision_available = torch.sum(labels, dim=1) > 0
    aggregate_mask = torch.where(torch.logical_and(is_pred_cell_selection, is_cell_supervision_available).view(aggregate_mask_init.size()), torch.zeros_like(aggregate_mask_init, dtype=torch.float32), aggregate_mask_init)
    aggregate_mask = aggregate_mask.detach()
    return aggregate_mask