import abc
from collections import namedtuple
from datetime import datetime
import pyrfc3339
from ._caveat import parse_caveat
from ._conditions import (
from ._declared import DECLARED_KEY
from ._namespace import Namespace
from ._operation import OP_KEY
from ._time import TIME_KEY
from ._utils import condition_with_prefix
def _check_allow(ctx, cond, arg):
    return _check_operations(ctx, True, arg)