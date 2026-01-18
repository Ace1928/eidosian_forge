from typing import (
from thinc.api import Model, Optimizer
from .compat import Protocol, runtime_checkable
@runtime_checkable
class TrainableComponent(Protocol):
    model: Any
    is_trainable: bool

    def update(self, examples: Iterable['Example'], *, drop: float=0.0, sgd: Optional[Optimizer]=None, losses: Optional[Dict[str, float]]=None) -> Dict[str, float]:
        ...

    def finish_update(self, sgd: Optimizer) -> None:
        ...