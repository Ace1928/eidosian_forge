import copy
import inspect
import re
from typing import (
from ..exceptions import InvalidOperationError
from ..utils.assertion import assert_or_throw
from ..utils.convert import get_full_type_path
from ..utils.entry_points import load_entry_point
from ..utils.hash import to_uuid
from .dict import IndexedOrderedDict
@FunctionWrapper.annotated_param('[Self]', '0', lambda a: False)
class SelfParam(AnnotatedParam):
    """For the self parameters in member functions"""
    pass