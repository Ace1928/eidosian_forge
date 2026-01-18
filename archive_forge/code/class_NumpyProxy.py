from __future__ import annotations
from typing import TYPE_CHECKING, Any
from typing_extensions import override
from .._utils import LazyProxy
from ._common import MissingDependencyError, format_instructions
class NumpyProxy(LazyProxy[Any]):

    @override
    def __load__(self) -> Any:
        try:
            import numpy
        except ImportError as err:
            raise MissingDependencyError(NUMPY_INSTRUCTIONS) from err
        return numpy