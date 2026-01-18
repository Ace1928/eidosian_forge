import operator
import re
import sqlalchemy as sa
from .. import config
from .. import engines
from .. import eq_
from .. import expect_raises
from .. import expect_raises_message
from .. import expect_warnings
from .. import fixtures
from .. import is_
from ..provision import get_temp_table_name
from ..provision import temp_table_keyword_args
from ..schema import Column
from ..schema import Table
from ... import event
from ... import ForeignKey
from ... import func
from ... import Identity
from ... import inspect
from ... import Integer
from ... import MetaData
from ... import String
from ... import testing
from ... import types as sql_types
from ...engine import Inspector
from ...engine import ObjectKind
from ...engine import ObjectScope
from ...exc import NoSuchTableError
from ...exc import UnreflectableTableError
from ...schema import DDL
from ...schema import Index
from ...sql.elements import quoted_name
from ...sql.schema import BLANK_SCHEMA
from ...testing import ComparesIndexes
from ...testing import ComparesTables
from ...testing import is_false
from ...testing import is_true
from ...testing import mock
def _resolve_names(self, schema, scope, filter_names, values):
    scope_filter = lambda _: True
    if scope is ObjectScope.DEFAULT:
        scope_filter = lambda k: 'tmp' not in k[1]
    if scope is ObjectScope.TEMPORARY:
        scope_filter = lambda k: 'tmp' in k[1]
    removed = {None: {'remote_table', 'remote_table_2'}, testing.config.test_schema: {'local_table', 'noncol_idx_test_nopk', 'noncol_idx_test_pk', 'user_tmp_v', self.temp_table_name()}}
    if not testing.requires.cross_schema_fk_reflection.enabled:
        removed[None].add('local_table')
        removed[testing.config.test_schema].update(['remote_table', 'remote_table_2'])
    if not testing.requires.index_reflection.enabled:
        removed[None].update(['noncol_idx_test_nopk', 'noncol_idx_test_pk'])
    if not testing.requires.temp_table_reflection.enabled or not testing.requires.temp_table_names.enabled:
        removed[None].update(['user_tmp_v', self.temp_table_name()])
    if not testing.requires.temporary_views.enabled:
        removed[None].update(['user_tmp_v'])
    res = {k: v for k, v in values.items() if scope_filter(k) and k[1] not in removed[schema] and (not filter_names or k[1] in filter_names)}
    return res