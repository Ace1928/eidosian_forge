from __future__ import annotations
from collections import defaultdict
import contextlib
from copy import copy
from itertools import filterfalse
import re
import sys
import warnings
from . import assertsql
from . import config
from . import engines
from . import mock
from .exclusions import db_spec
from .util import fail
from .. import exc as sa_exc
from .. import schema
from .. import sql
from .. import types as sqltypes
from .. import util
from ..engine import default
from ..engine import url
from ..sql.selectable import LABEL_STYLE_TABLENAME_PLUS_COL
from ..util import decorator
def _assert_proper_exception_context(exception):
    """assert that any exception we're catching does not have a __context__
    without a __cause__, and that __suppress_context__ is never set.

    Python 3 will report nested as exceptions as "during the handling of
    error X, error Y occurred". That's not what we want to do.  we want
    these exceptions in a cause chain.

    """
    if exception.__context__ is not exception.__cause__ and (not exception.__suppress_context__):
        assert False, 'Exception %r was correctly raised but did not set a cause, within context %r as its cause.' % (exception, exception.__context__)