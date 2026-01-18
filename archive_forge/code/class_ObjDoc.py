import inspect
import textwrap
import re
import pydoc
from warnings import warn
from collections import namedtuple
from collections.abc import Callable, Mapping
import copy
import sys
class ObjDoc(NumpyDocString):

    def __init__(self, obj, doc=None, config=None):
        self._f = obj
        if config is None:
            config = {}
        NumpyDocString.__init__(self, doc, config=config)