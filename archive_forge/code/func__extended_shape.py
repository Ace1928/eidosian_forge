import warnings
from typing import Any, Dict, Optional, Tuple
import torch
from torch.distributions import constraints
from torch.distributions.utils import lazy_property
from torch.types import _size
def _extended_shape(self, sample_shape: _size=torch.Size()) -> Tuple[int, ...]:
    """
        Returns the size of the sample returned by the distribution, given
        a `sample_shape`. Note, that the batch and event shapes of a distribution
        instance are fixed at the time of construction. If this is empty, the
        returned shape is upcast to (1,).

        Args:
            sample_shape (torch.Size): the size of the sample to be drawn.
        """
    if not isinstance(sample_shape, torch.Size):
        sample_shape = torch.Size(sample_shape)
    return torch.Size(sample_shape + self._batch_shape + self._event_shape)