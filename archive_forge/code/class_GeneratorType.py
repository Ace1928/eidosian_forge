import math
from typing import Literal, Optional, Tuple, Union
import torch
from torch import Tensor, nn
from torchmetrics.functional.image.lpips import _LPIPS
from torchmetrics.utilities.imports import _TORCHVISION_AVAILABLE
class GeneratorType(nn.Module):
    """Basic interface for a generator model.

    Users can inherit from this class and implement their own generator model. The requirements are that the ``sample``
    method is implemented and that the ``num_classes`` attribute is present when ``conditional=True`` metric.

    """

    @property
    def num_classes(self) -> int:
        """Return the number of classes for conditional generation."""
        raise NotImplementedError

    def sample(self, num_samples: int) -> Tensor:
        """Sample from the generator.

        Args:
            num_samples: Number of samples to generate.

        """
        raise NotImplementedError