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
class ScopeDecorator:

    def __init__(self, scope: Type[Scope]) -> None:
        self.scope = scope

    def __call__(self, cls: T) -> T:
        cast(Any, cls).__scope__ = self.scope
        binding = getattr(cls, '__binding__', None)
        if binding:
            new_binding = Binding(interface=binding.interface, provider=binding.provider, scope=self.scope)
            setattr(cls, '__binding__', new_binding)
        return cls

    def __repr__(self) -> str:
        return 'ScopeDecorator(%s)' % self.scope.__name__