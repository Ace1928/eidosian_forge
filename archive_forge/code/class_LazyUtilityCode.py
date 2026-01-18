from __future__ import absolute_import
import cython
import hashlib
import operator
import os
import re
import shutil
import textwrap
from string import Template
from functools import partial
from contextlib import closing, contextmanager
from collections import defaultdict
from . import Naming
from . import Options
from . import DebugFlags
from . import StringEncoding
from .. import Utils
from .Scanning import SourceDescriptor
from ..StringIOTree import StringIOTree
class LazyUtilityCode(UtilityCodeBase):
    """
    Utility code that calls a callback with the root code writer when
    available. Useful when you only have 'env' but not 'code'.
    """
    __name__ = '<lazy>'
    requires = None

    def __init__(self, callback):
        self.callback = callback

    def put_code(self, globalstate):
        utility = self.callback(globalstate.rootwriter)
        globalstate.use_utility_code(utility)