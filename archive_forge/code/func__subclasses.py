import collections
import contextlib
import copy
import inspect
import json
import sys
import textwrap
from typing import (
from itertools import zip_longest
from importlib.metadata import version as importlib_version
from typing import Final
import jsonschema
import jsonschema.exceptions
import jsonschema.validators
import numpy as np
import pandas as pd
from packaging.version import Version
from altair import vegalite
def _subclasses(cls: type) -> Generator[type, None, None]:
    """Breadth-first sequence of all classes which inherit from cls."""
    seen = set()
    current_set = {cls}
    while current_set:
        seen |= current_set
        current_set = set.union(*(set(cls.__subclasses__()) for cls in current_set))
        for cls in current_set - seen:
            yield cls