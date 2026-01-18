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
class CheckCompilerAccess:

    def __init__(self, test_statement):
        self.test_statement = test_statement
        self._annotations = {}
        self.supports_execution = getattr(test_statement, 'supports_execution', False)
        if self.supports_execution:
            self._execution_options = test_statement._execution_options
            if hasattr(test_statement, '_returning'):
                self._returning = test_statement._returning
            if hasattr(test_statement, '_inline'):
                self._inline = test_statement._inline
            if hasattr(test_statement, '_return_defaults'):
                self._return_defaults = test_statement._return_defaults

    @property
    def _variant_mapping(self):
        return self.test_statement._variant_mapping

    def _default_dialect(self):
        return self.test_statement._default_dialect()

    def compile(self, dialect, **kw):
        return self.test_statement.compile.__func__(self, dialect=dialect, **kw)

    def _compiler(self, dialect, **kw):
        return self.test_statement._compiler.__func__(self, dialect, **kw)

    def _compiler_dispatch(self, compiler, **kwargs):
        if hasattr(compiler, 'statement'):
            with mock.patch.object(compiler, 'statement', DontAccess()):
                return self.test_statement._compiler_dispatch(compiler, **kwargs)
        else:
            return self.test_statement._compiler_dispatch(compiler, **kwargs)