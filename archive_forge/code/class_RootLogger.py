import sys, os, time, io, re, traceback, warnings, weakref, collections.abc
from types import GenericAlias
from string import Template
from string import Formatter as StrFormatter
import threading
import atexit
class RootLogger(Logger):
    """
    A root logger is not that different to any other logger, except that
    it must have a logging level and there is only one instance of it in
    the hierarchy.
    """

    def __init__(self, level):
        """
        Initialize the logger with the name "root".
        """
        Logger.__init__(self, 'root', level)

    def __reduce__(self):
        return (getLogger, ())