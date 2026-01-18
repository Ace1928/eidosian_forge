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
class ComparesIndexes:

    def compare_table_index_with_expected(self, table: schema.Table, expected: list, dialect_name: str):
        eq_(len(table.indexes), len(expected))
        idx_dict = {idx.name: idx for idx in table.indexes}
        for exp in expected:
            idx = idx_dict[exp['name']]
            eq_(idx.unique, exp['unique'])
            cols = [c for c in exp['column_names'] if c is not None]
            eq_(len(idx.columns), len(cols))
            for c in cols:
                is_true(c in idx.columns)
            exprs = exp.get('expressions')
            if exprs:
                eq_(len(idx.expressions), len(exprs))
                for idx_exp, expr, col in zip(idx.expressions, exprs, exp['column_names']):
                    if col is None:
                        eq_(idx_exp.text, expr)
            if exp.get('dialect_options') and f'{dialect_name}_include' in exp['dialect_options']:
                eq_(idx.dialect_options[dialect_name]['include'], exp['dialect_options'][f'{dialect_name}_include'])