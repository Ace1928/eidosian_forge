import inspect
import re
from collections import defaultdict
from enum import Enum
from typing import (
from thinc.api import ConfigValidationError, Model, Optimizer
from thinc.config import Promise
from .attrs import NAMES
from .compat import Literal
from .lookups import Lookups
from .util import is_cython_func
class TokenPatternOperatorMinMax(ConstrainedStr):
    regex = re.compile('^({\\d+}|{\\d+,\\d*}|{\\d*,\\d+})$')