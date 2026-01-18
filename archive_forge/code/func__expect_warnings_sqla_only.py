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
def _expect_warnings_sqla_only(exc_cls, messages, regex=True, search_msg=False, assert_=True):
    """SQLAlchemy internal use only _expect_warnings().

    Alembic is using _expect_warnings() directly, and should be updated
    to use this new interface.

    """
    return _expect_warnings(exc_cls, messages, regex=regex, search_msg=search_msg, assert_=assert_, raise_on_any_unexpected=True)