from __future__ import print_function
import re
import sys
import time
from pygments.filter import apply_filters, Filter
from pygments.filters import get_filter_by_name
from pygments.token import Error, Text, Other, _TokenType
from pygments.util import get_bool_opt, get_int_opt, get_list_opt, \
from pygments.regexopt import regex_opt
class default:
    """
    Indicates a state or state action (e.g. #pop) to apply.
    For example default('#pop') is equivalent to ('', Token, '#pop')
    Note that state tuples may be used as well.

    .. versionadded:: 2.0
    """

    def __init__(self, state):
        self.state = state