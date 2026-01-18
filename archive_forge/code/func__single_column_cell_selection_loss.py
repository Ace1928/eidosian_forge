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
def _single_column_cell_selection_loss(token_logits, column_logits, labels, cell_index, col_index, cell_mask):
    """
    Computes the loss for cell selection constrained to a single column. The loss is a hierarchical log-likelihood. The
    model first predicts a column and then selects cells within that column (conditioned on the column). Cells outside
    the selected column are never selected.

    Args:
        token_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Tensor containing the logits per token.
        column_logits (`torch.FloatTensor` of shape `(batch_size, max_num_cols)`):
            Tensor containing the logits per column.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Labels per token.
        cell_index (`ProductIndexMap`):
            Index that groups tokens into cells.
        col_index (`IndexMap`):
            Index that groups tokens into columns.
        cell_mask (`torch.FloatTensor` of shape `(batch_size, max_num_rows * max_num_cols)`):
            Mask for cells that exist in the table (i.e. that are not padding).

    Returns:
        selection_loss_per_example (`torch.FloatTensor` of shape `(batch_size,)`): Loss for each example. logits
        (`torch.FloatTensor` of shape `(batch_size, sequence_length)`): New logits which are only allowed to select
        cells in a single column. Logits outside of the most likely column according to *column_logits* will be set to
        a very low value (such that the probabilities are 0).
    """
    labels_per_column, _ = reduce_sum(torch.as_tensor(labels, dtype=torch.float32, device=labels.device), col_index)
    column_label = torch.argmax(labels_per_column, dim=-1)
    no_cell_selected = torch.eq(torch.max(labels_per_column, dim=-1)[0], 0)
    column_label = torch.where(no_cell_selected.view(column_label.size()), torch.zeros_like(column_label), column_label)
    column_dist = torch.distributions.Categorical(logits=column_logits)
    column_loss_per_example = -column_dist.log_prob(column_label)
    logits_per_cell, _ = reduce_mean(token_logits, cell_index)
    labels_per_cell, labels_index = reduce_max(torch.as_tensor(labels, dtype=torch.long, device=labels.device), cell_index)
    column_id_for_cells = cell_index.project_inner(labels_index).indices
    column_mask = torch.as_tensor(torch.eq(column_id_for_cells, torch.unsqueeze(column_label, dim=-1)), dtype=torch.float32, device=cell_mask.device)
    cell_dist = torch.distributions.Bernoulli(logits=logits_per_cell)
    cell_log_prob = cell_dist.log_prob(labels_per_cell.type(torch.float32))
    cell_loss = -torch.sum(cell_log_prob * column_mask * cell_mask, dim=1)
    cell_loss /= torch.sum(column_mask * cell_mask, dim=1) + EPSILON_ZERO_DIVISION
    selection_loss_per_example = column_loss_per_example
    selection_loss_per_example += torch.where(no_cell_selected.view(selection_loss_per_example.size()), torch.zeros_like(selection_loss_per_example), cell_loss)
    selected_column_id = torch.as_tensor(torch.argmax(column_logits, dim=-1), dtype=torch.long, device=column_logits.device)
    selected_column_mask = torch.as_tensor(torch.eq(column_id_for_cells, torch.unsqueeze(selected_column_id, dim=-1)), dtype=torch.float32, device=selected_column_id.device)
    selected_column_mask = torch.where(torch.eq(column_id_for_cells, 0).view(selected_column_mask.size()), torch.zeros_like(selected_column_mask), selected_column_mask)
    new_logits_per_cell = logits_per_cell + CLOSE_ENOUGH_TO_LOG_ZERO * (1.0 - cell_mask * selected_column_mask)
    logits = gather(new_logits_per_cell, cell_index)
    return (selection_loss_per_example, logits)