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
def _valid(out):
    if out.allow_duplicate and (not prevent_initial_call) and (not config_prevent_initial_call):
        raise exceptions.DuplicateCallback("allow_duplicate requires prevent_initial_call to be True. The order of the call is not guaranteed to be the same on every page load. To enable duplicate callback with initial call, set prevent_initial_call='initial_duplicate'  or globally in the config prevent_initial_callbacks='initial_duplicate'")