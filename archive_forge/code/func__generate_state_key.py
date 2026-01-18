from typing import Any, Dict, Type
from torch import Tensor
from torch.optim import Optimizer
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
def _generate_state_key(self, **kwargs: Any) -> str:
    """Formats a set of key-value pairs into a state key string with the callback class name prefixed. Useful for
        defining a :attr:`state_key`.

        Args:
            **kwargs: A set of key-value pairs. Must be serializable to :class:`str`.

        """
    return f'{self.__class__.__qualname__}{repr(kwargs)}'