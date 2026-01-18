import sys, os, time, io, re, traceback, warnings, weakref, collections.abc
from types import GenericAlias
from string import Template
from string import Formatter as StrFormatter
import threading
import atexit
def _fixupParents(self, alogger):
    """
        Ensure that there are either loggers or placeholders all the way
        from the specified logger to the root of the logger hierarchy.
        """
    name = alogger.name
    i = name.rfind('.')
    rv = None
    while i > 0 and (not rv):
        substr = name[:i]
        if substr not in self.loggerDict:
            self.loggerDict[substr] = PlaceHolder(alogger)
        else:
            obj = self.loggerDict[substr]
            if isinstance(obj, Logger):
                rv = obj
            else:
                assert isinstance(obj, PlaceHolder)
                obj.append(alogger)
        i = name.rfind('.', 0, i - 1)
    if not rv:
        rv = self.root
    alogger.parent = rv