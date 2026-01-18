import ast
import dataclasses
import inspect
import os
from functools import partial
from typing import Callable, Dict, List
from torch._jit_internal import FAKE_FILENAME_PREFIX, is_optional
from torch._sources import ParsedDef, SourceContext
def _get_fake_filename(cls, method_name):
    return os.path.join(FAKE_FILENAME_PREFIX, cls.__name__, method_name)