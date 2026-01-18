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
def _calculate_beta_divergence(self, preds_distribution: Tensor, target_distribution: Tensor) -> Tensor:
    """Calculate beta divergence between discrete distributions of predicted and reference sentences.

        Args:
            preds_distribution:
                Discrete reference distribution of predicted sentences over the vocabulary.
            target_distribution:
                Discrete reference distribution of reference sentences over the vocabulary.

        Return:
            Beta divergence between discrete distributions of predicted and reference sentences.

        """
    self.alpha = 1.0
    return self._calculate_ab_divergence(preds_distribution, target_distribution)