import sys, os, time, io, re, traceback, warnings, weakref, collections.abc
from types import GenericAlias
from string import Template
from string import Formatter as StrFormatter
import threading
import atexit
def addFilter(self, filter):
    """
        Add the specified filter to this handler.
        """
    if not filter in self.filters:
        self.filters.append(filter)