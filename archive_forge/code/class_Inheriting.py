from __future__ import annotations
from collections.abc import Hashable, Mapping, Sequence
from typing import Any
import pytest
import dask
import dask.threaded
from dask.base import DaskMethodsMixin, dont_optimize, tokenize
from dask.context import globalmethod
from dask.delayed import Delayed, delayed
from dask.typing import (
class Inheriting(DaskCollection):

    def __init__(self, based_on: DaskCollection) -> None:
        self.based_on = based_on

    def __dask_graph__(self) -> Graph:
        return self.based_on.__dask_graph__()

    def __dask_keys__(self) -> NestedKeys:
        return self.based_on.__dask_keys__()

    def __dask_postcompute__(self) -> tuple[PostComputeCallable, tuple]:
        return (finalize, ())

    def __dask_postpersist__(self) -> tuple[PostPersistCallable, tuple]:
        return self.based_on.__dask_postpersist__()

    def __dask_tokenize__(self) -> Hashable:
        return tokenize(self.based_on)
    __dask_scheduler__ = staticmethod(dask.threaded.get)
    __dask_optimize__ = globalmethod(dont_optimize, key='hlgcollection_optim', falsey=dont_optimize)

    def compute(self, **kwargs) -> Any:
        return dask.compute(self, **kwargs)

    def persist(self, **kwargs) -> Inheriting:
        return Inheriting(self.based_on.persist(**kwargs))

    def visualize(self, filename: str='mydask', format: str | None=None, optimize_graph: bool=False, **kwargs: Any) -> DisplayObject | None:
        return dask.visualize(self, filename=filename, format=format, optimize_graph=optimize_graph, **kwargs)