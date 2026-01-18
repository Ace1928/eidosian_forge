import os
from enum import unique
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: N812
from torch.utils.data import DataLoader
from typing_extensions import Literal
from torchmetrics.functional.text.helper_embedding_metric import (
from torchmetrics.utilities.enums import EnumStr
from torchmetrics.utilities.imports import _TRANSFORMERS_GREATER_EQUAL_4_4
def _calculate_renyi_divergence(self, preds_distribution: Tensor, target_distribution: Tensor) -> Tensor:
    """Calculate Rényi divergence between discrete distributions of predicted and reference sentences.

        Args:
            preds_distribution:
                Discrete reference distribution of predicted sentences over the vocabulary.
            target_distribution:
                Discrete reference distribution of reference sentences over the vocabulary.

        Return:
            Rényi divergence between discrete distributions of predicted and reference sentences.

        """
    return torch.log(torch.sum(target_distribution ** self.alpha * preds_distribution ** (1 - self.alpha), dim=-1)) / (self.alpha - 1)