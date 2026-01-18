import datetime
from apitools.gen import message_registry
from apitools.gen import service_registry
from apitools.gen import util
def WriteIntermediateInit(self, out):
    """Write a simple __init__.py for an intermediate directory."""
    printer = self._GetPrinter(out)
    printer('#!/usr/bin/env python')
    printer('"""Shared __init__.py for apitools."""')
    printer()
    printer('from pkgutil import extend_path')
    printer('__path__ = extend_path(__path__, __name__)')