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
class MapBindProvider(ListOfProviders[Dict[str, T]]):
    """A provider for map bindings."""

    def get(self, injector: 'Injector') -> Dict[str, T]:
        map: Dict[str, T] = {}
        for provider in self._providers:
            map.update(provider.get(injector))
        return map