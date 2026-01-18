import sys
from collections.abc import MutableSequence
import re
from textwrap import dedent
from keyword import iskeyword
import flask
from ._grouping import grouping_len, map_grouping
from .development.base_component import Component
from . import exceptions
from ._utils import (
def _can_serialize(val):
    if not (_valid_child(val) or _valid_prop(val)):
        return False
    try:
        to_json(val)
    except TypeError:
        return False
    return True