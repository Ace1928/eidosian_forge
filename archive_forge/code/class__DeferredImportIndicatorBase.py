from collections.abc import Mapping
import inspect
import importlib
import logging
import sys
import warnings
from .deprecation import deprecated, deprecation_warning, in_testing_environment
from .errors import DeferredImportError
class _DeferredImportIndicatorBase(object):

    def __and__(self, other):
        return _DeferredAnd(self, other)

    def __or__(self, other):
        return _DeferredOr(self, other)

    def __rand__(self, other):
        return _DeferredAnd(other, self)

    def __ror__(self, other):
        return _DeferredOr(other, self)