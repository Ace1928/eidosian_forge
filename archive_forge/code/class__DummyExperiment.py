from abc import ABC, abstractmethod
from argparse import Namespace
from functools import wraps
from typing import Any, Callable, Dict, Optional, Union
from torch import Tensor
from torch.nn import Module
from lightning_fabric.utilities.rank_zero import rank_zero_only
class _DummyExperiment:
    """Dummy experiment."""

    def nop(self, *args: Any, **kw: Any) -> None:
        pass

    def __getattr__(self, _: Any) -> Callable:
        return self.nop

    def __getitem__(self, idx: int) -> '_DummyExperiment':
        return self

    def __setitem__(self, *args: Any, **kwargs: Any) -> None:
        pass