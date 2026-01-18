from sys import version_info as _swig_python_version_info
import weakref
import inspect
import os
import re
import sys
import traceback
import inspect
import io
import os
import sys
import traceback
import types
class IteratorWrap:
    """
    This is a Python iterator for containers that have C++-style
    begin() and end() methods that return iterators.

    Iterators must have the following methods:

        __increment__(): move to next item in the container.
        __ref__(): return reference to item in the container.

    Must also be able to compare two iterators for equality.

    """

    def __init__(self, container):
        self.container = container
        self.pos = None
        self.end = container.end()

    def __iter__(self):
        return self

    def __next__(self):
        if self.pos is None:
            self.pos = self.container.begin()
        else:
            self.pos.__increment__()
        if self.pos == self.end:
            raise StopIteration()
        return self.pos.__ref__()

    def next(self):
        return self.__next__()