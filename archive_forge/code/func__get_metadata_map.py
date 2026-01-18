from collections import namedtuple
import inspect
import re
import numpy as np
import math
from textwrap import dedent
import unittest
import warnings
from numba.tests.support import (TestCase, override_config,
from numba import jit, njit
from numba.core import types
from numba.core.datamodel import default_manager
from numba.core.errors import NumbaDebugInfoWarning
import llvmlite.binding as llvm
def _get_metadata_map(self, metadata):
    """Gets the map of DI label to md, e.g.
        '!33' -> '!{!"branch_weights", i32 1, i32 99}'
        """
    metadata_definition_map = dict()
    meta_definition_split = re.compile('(![0-9]+) = (.*)')
    for line in metadata:
        matched = meta_definition_split.match(line)
        if matched:
            dbg_val, info = matched.groups()
            metadata_definition_map[dbg_val] = info
    return metadata_definition_map