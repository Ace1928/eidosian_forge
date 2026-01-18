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
@contextlib.contextmanager
def _expect_raises(except_cls, msg=None, check_context=False):
    if isinstance(except_cls, type) and issubclass(except_cls, Warning) or isinstance(except_cls, Warning):
        raise TypeError('Use expect_warnings for warnings, not expect_raises / assert_raises')
    ec = _ErrorContainer()
    if check_context:
        are_we_already_in_a_traceback = sys.exc_info()[0]
    try:
        yield ec
        success = False
    except except_cls as err:
        ec.error = err
        success = True
        if msg is not None:
            error_as_string = str(err)
            assert re.search(msg, error_as_string, re.UNICODE), '%r !~ %s' % (msg, error_as_string)
        if check_context and (not are_we_already_in_a_traceback):
            _assert_proper_exception_context(err)
        print(str(err).encode('utf-8'))
    del ec
    assert success, 'Callable did not raise an exception'