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
class ComputedReflectionTest(fixtures.ComputedReflectionFixtureTest):

    def test_computed_col_default_not_set(self):
        insp = inspect(config.db)
        cols = insp.get_columns('computed_default_table')
        col_data = {c['name']: c for c in cols}
        is_true('42' in col_data['with_default']['default'])
        is_(col_data['normal']['default'], None)
        is_(col_data['computed_col']['default'], None)

    def test_get_column_returns_computed(self):
        insp = inspect(config.db)
        cols = insp.get_columns('computed_default_table')
        data = {c['name']: c for c in cols}
        for key in ('id', 'normal', 'with_default'):
            is_true('computed' not in data[key])
        compData = data['computed_col']
        is_true('computed' in compData)
        is_true('sqltext' in compData['computed'])
        eq_(self.normalize(compData['computed']['sqltext']), 'normal+42')
        eq_('persisted' in compData['computed'], testing.requires.computed_columns_reflect_persisted.enabled)
        if testing.requires.computed_columns_reflect_persisted.enabled:
            eq_(compData['computed']['persisted'], testing.requires.computed_columns_default_persisted.enabled)

    def check_column(self, data, column, sqltext, persisted):
        is_true('computed' in data[column])
        compData = data[column]['computed']
        eq_(self.normalize(compData['sqltext']), sqltext)
        if testing.requires.computed_columns_reflect_persisted.enabled:
            is_true('persisted' in compData)
            is_(compData['persisted'], persisted)

    def test_get_column_returns_persisted(self):
        insp = inspect(config.db)
        cols = insp.get_columns('computed_column_table')
        data = {c['name']: c for c in cols}
        self.check_column(data, 'computed_no_flag', 'normal+42', testing.requires.computed_columns_default_persisted.enabled)
        if testing.requires.computed_columns_virtual.enabled:
            self.check_column(data, 'computed_virtual', 'normal+2', False)
        if testing.requires.computed_columns_stored.enabled:
            self.check_column(data, 'computed_stored', 'normal-42', True)

    @testing.requires.schemas
    def test_get_column_returns_persisted_with_schema(self):
        insp = inspect(config.db)
        cols = insp.get_columns('computed_column_table', schema=config.test_schema)
        data = {c['name']: c for c in cols}
        self.check_column(data, 'computed_no_flag', 'normal/42', testing.requires.computed_columns_default_persisted.enabled)
        if testing.requires.computed_columns_virtual.enabled:
            self.check_column(data, 'computed_virtual', 'normal/2', False)
        if testing.requires.computed_columns_stored.enabled:
            self.check_column(data, 'computed_stored', 'normal*42', True)