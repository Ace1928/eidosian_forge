from contextlib import contextmanager
from dataclasses import fields
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union, overload
from weakref import proxy
import torch
from torch import optim
from torch.optim import Optimizer
from typing_extensions import override
import pytorch_lightning as pl
from lightning_fabric.utilities.types import Optimizable, ReduceLROnPlateau, _Stateful
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
from pytorch_lightning.utilities.signature_utils import is_param_in_hook_signature
from pytorch_lightning.utilities.types import LRSchedulerConfig, LRSchedulerTypeTuple
class _MockOptimizer(Optimizer):
    """The `_MockOptimizer` will be used inplace of an optimizer in the event that `None` is returned from
    :meth:`~pytorch_lightning.core.LightningModule.configure_optimizers`."""

    def __init__(self) -> None:
        super().__init__([torch.zeros(1)], {})

    @override
    def add_param_group(self, param_group: Dict[Any, Any]) -> None:
        pass

    @override
    def load_state_dict(self, state_dict: Dict[Any, Any]) -> None:
        pass

    @override
    def state_dict(self) -> Dict[str, Any]:
        return {}

    @overload
    def step(self, closure: None=...) -> None:
        ...

    @overload
    def step(self, closure: Callable[[], float]) -> float:
        ...

    @override
    def step(self, closure: Optional[Callable[[], float]]=None) -> Optional[float]:
        if closure is not None:
            return closure()

    @override
    def zero_grad(self, set_to_none: Optional[bool]=True) -> None:
        pass

    @override
    def __repr__(self) -> str:
        return 'No Optimizer'