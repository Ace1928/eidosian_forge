import inspect
import types
import traceback
import sys
import operator as op
from collections import namedtuple
import warnings
import re
import builtins
import typing
from pathlib import Path
from typing import Optional, Tuple
from jedi.inference.compiled.getattr_static import getattr_static
def _annotation_to_str(self, annotation):
    return inspect.formatannotation(annotation)