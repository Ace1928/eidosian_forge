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
def _expect_warnings(exc_cls, messages, regex=True, search_msg=False, assert_=True, raise_on_any_unexpected=False, squelch_other_warnings=False):
    global _FILTERS, _SEEN, _EXC_CLS
    if regex or search_msg:
        filters = [re.compile(msg, re.I | re.S) for msg in messages]
    else:
        filters = list(messages)
    if _FILTERS is not None:
        assert _SEEN is not None
        assert _EXC_CLS is not None
        _FILTERS.extend(filters)
        _SEEN.update(filters)
        _EXC_CLS += (exc_cls,)
        yield
    else:
        seen = _SEEN = set(filters)
        _FILTERS = filters
        _EXC_CLS = (exc_cls,)
        if raise_on_any_unexpected:

            def real_warn(msg, *arg, **kw):
                raise AssertionError('Got unexpected warning: %r' % msg)
        else:
            real_warn = warnings.warn

        def our_warn(msg, *arg, **kw):
            if isinstance(msg, _EXC_CLS):
                exception = type(msg)
                msg = str(msg)
            elif arg:
                exception = arg[0]
            else:
                exception = None
            if not exception or not issubclass(exception, _EXC_CLS):
                if not squelch_other_warnings:
                    return real_warn(msg, *arg, **kw)
                else:
                    return
            if not filters and (not raise_on_any_unexpected):
                return
            for filter_ in filters:
                if search_msg and filter_.search(msg) or (regex and filter_.match(msg)) or (not regex and filter_ == msg):
                    seen.discard(filter_)
                    break
            else:
                if not squelch_other_warnings:
                    real_warn(msg, *arg, **kw)
        with mock.patch('warnings.warn', our_warn):
            try:
                yield
            finally:
                _SEEN = _FILTERS = _EXC_CLS = None
                if assert_:
                    assert not seen, 'Warnings were not seen: %s' % ', '.join(('%r' % (s.pattern if regex else s) for s in seen))