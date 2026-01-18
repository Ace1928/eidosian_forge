import contextlib
import copy
from abc import ABC, abstractmethod
from typing import (
import torch.nn as nn
class _ConfigAutoWrap:
    """
    Helper class to wrap modules based on default config args via a context manager.
    See :func:`enable_wrap` for more information.
    """
    in_autowrap_context: bool = False
    wrapper_cls: Optional[Callable] = None
    kwargs: Dict[str, Any] = {}

    def __init__(self, **kwargs: Dict[str, Any]):
        self.kwargs = kwargs

    @staticmethod
    def enable_autowrap_context(kwargs: Any) -> None:
        if _ConfigAutoWrap.in_autowrap_context:
            raise NotImplementedError('You are already within an autowrap context and we currently do not supported nested autowrap.')
        _ConfigAutoWrap.in_autowrap_context = True
        assert 'wrapper_cls' in kwargs.keys(), 'Expected to pass in wrapper_cls arg into _ConfigAutoWrap.'
        _ConfigAutoWrap.wrapper_cls = cast(Callable, kwargs['wrapper_cls'])
        del kwargs['wrapper_cls']
        _ConfigAutoWrap.kwargs = kwargs

    @staticmethod
    def disable_autowrap_context() -> None:
        _ConfigAutoWrap.in_autowrap_context = False
        _ConfigAutoWrap.wrapper_cls = None
        _ConfigAutoWrap.kwargs = {}

    def __enter__(self) -> None:
        self.enable_autowrap_context(self.kwargs)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.disable_autowrap_context()