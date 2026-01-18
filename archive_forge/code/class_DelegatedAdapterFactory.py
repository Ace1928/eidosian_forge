import sys
from functools import partial
from inspect import isclass
from threading import Lock, RLock
from .arguments import formatargspec
from .__wrapt__ import (FunctionWrapper, BoundFunctionWrapper, ObjectProxy,
class DelegatedAdapterFactory(AdapterFactory):

    def __init__(self, factory):
        super(DelegatedAdapterFactory, self).__init__()
        self.factory = factory

    def __call__(self, wrapped):
        return self.factory(wrapped)