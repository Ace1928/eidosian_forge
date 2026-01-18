from abc import ABC, abstractmethod
import torch
from torch import Tensor
from torch.nn import Module
from typing_extensions import override
class LayerSync(ABC):
    """Abstract base class for creating plugins that wrap layers of a model with synchronization logic for
    multiprocessing."""

    @abstractmethod
    def apply(self, model: Module) -> Module:
        """Override this method to apply synchronization to the layers of this model."""

    @abstractmethod
    def revert(self, model: Module) -> Module:
        """Override this method to undo all modifications made in :meth:`apply`."""