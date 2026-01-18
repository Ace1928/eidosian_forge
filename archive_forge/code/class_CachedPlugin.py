import os
import pickle
import sys
import types
from typing import Iterable, Optional, Type, TypeVar
from zope.interface import Interface, providedBy
from twisted.python import log
from twisted.python.components import getAdapterFactory
from twisted.python.modules import getModule
from twisted.python.reflect import namedAny
class CachedPlugin:

    def __init__(self, dropin, name, description, provided):
        self.dropin = dropin
        self.name = name
        self.description = description
        self.provided = provided
        self.dropin.plugins.append(self)

    def __repr__(self) -> str:
        return '<CachedPlugin {!r}/{!r} (provides {!r})>'.format(self.name, self.dropin.moduleName, ', '.join([i.__name__ for i in self.provided]))

    def load(self):
        return namedAny(self.dropin.moduleName + '.' + self.name)

    def __conform__(self, interface, registry=None, default=None):
        for providedInterface in self.provided:
            if providedInterface.isOrExtends(interface):
                return self.load()
            if getAdapterFactory(providedInterface, interface, None) is not None:
                return interface(self.load(), default)
        return default
    getComponent = __conform__