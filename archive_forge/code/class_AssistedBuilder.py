import functools
import inspect
import itertools
import logging
import sys
import threading
import types
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from typing import (
class AssistedBuilder(Generic[T]):

    def __init__(self, injector: Injector, target: Type[T]) -> None:
        self._injector = injector
        self._target = target

    def build(self, **kwargs: Any) -> T:
        binder = self._injector.binder
        binding, _ = binder.get_binding(self._target)
        provider = binding.provider
        if not isinstance(provider, ClassProvider):
            raise Error('Assisted interface building works only with ClassProviders, got %r for %r' % (provider, binding.interface))
        return self._build_class(cast(Type[T], provider._cls), **kwargs)

    def _build_class(self, cls: Type[T], **kwargs: Any) -> T:
        return self._injector.create_object(cls, additional_kwargs=kwargs)