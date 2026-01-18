from __future__ import division
import pyparsing as pp
from collections import namedtuple
from datetime import datetime
def _get_decl_line_no(cls):
    import inspect
    return inspect.getsourcelines(cls)[1]