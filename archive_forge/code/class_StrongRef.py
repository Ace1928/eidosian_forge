imported with ``from foo import ...`` was also updated.
from IPython.core import magic_arguments
from IPython.core.magic import Magics, magics_class, line_magic
import os
import sys
import traceback
import types
import weakref
import gc
import logging
from importlib import import_module, reload
from importlib.util import source_from_cache
class StrongRef:

    def __init__(self, obj):
        self.obj = obj

    def __call__(self):
        return self.obj