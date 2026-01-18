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
class ComponentReflectionTestExtra(ComparesIndexes, fixtures.TestBase):
    __backend__ = True

    @testing.combinations((True, testing.requires.schemas), (False,), argnames='use_schema')
    @testing.requires.check_constraint_reflection
    def test_get_check_constraints(self, metadata, connection, use_schema):
        if use_schema:
            schema = config.test_schema
        else:
            schema = None
        Table('sa_cc', metadata, Column('a', Integer()), sa.CheckConstraint('a > 1 AND a < 5', name='cc1'), sa.CheckConstraint('a = 1 OR (a > 2 AND a < 5)', name='UsesCasing'), schema=schema)
        Table('no_constraints', metadata, Column('data', sa.String(20)), schema=schema)
        metadata.create_all(connection)
        insp = inspect(connection)
        reflected = sorted(insp.get_check_constraints('sa_cc', schema=schema), key=operator.itemgetter('name'))

        def normalize(sqltext):
            return ' '.join(re.findall('and|\\d|=|a|or|<|>', sqltext.lower(), re.I))
        reflected = [{'name': item['name'], 'sqltext': normalize(item['sqltext'])} for item in reflected]
        eq_(reflected, [{'name': 'UsesCasing', 'sqltext': 'a = 1 or a > 2 and a < 5'}, {'name': 'cc1', 'sqltext': 'a > 1 and a < 5'}])
        no_cst = 'no_constraints'
        eq_(insp.get_check_constraints(no_cst, schema=schema), [])

    @testing.requires.indexes_with_expressions
    def test_reflect_expression_based_indexes(self, metadata, connection):
        t = Table('t', metadata, Column('x', String(30)), Column('y', String(30)), Column('z', String(30)))
        Index('t_idx', func.lower(t.c.x), t.c.z, func.lower(t.c.y))
        long_str = 'long string ' * 100
        Index('t_idx_long', func.coalesce(t.c.x, long_str))
        Index('t_idx_2', t.c.x)
        metadata.create_all(connection)
        insp = inspect(connection)
        expected = [{'name': 't_idx_2', 'column_names': ['x'], 'unique': False, 'dialect_options': {}}]

        def completeIndex(entry):
            if testing.requires.index_reflects_included_columns.enabled:
                entry['include_columns'] = []
                entry['dialect_options'] = {f'{connection.engine.name}_include': []}
            else:
                entry.setdefault('dialect_options', {})
        completeIndex(expected[0])

        class lower_index_str(str):

            def __eq__(self, other):
                ol = other.lower()
                return 'lower' in ol and ('x' in ol or 'y' in ol)

        class coalesce_index_str(str):

            def __eq__(self, other):
                return 'coalesce' in other.lower() and long_str in other
        if testing.requires.reflect_indexes_with_expressions.enabled:
            expr_index = {'name': 't_idx', 'column_names': [None, 'z', None], 'expressions': [lower_index_str('lower(x)'), 'z', lower_index_str('lower(y)')], 'unique': False}
            completeIndex(expr_index)
            expected.insert(0, expr_index)
            expr_index_long = {'name': 't_idx_long', 'column_names': [None], 'expressions': [coalesce_index_str(f"coalesce(x, '{long_str}')")], 'unique': False}
            completeIndex(expr_index_long)
            expected.append(expr_index_long)
            eq_(insp.get_indexes('t'), expected)
            m2 = MetaData()
            t2 = Table('t', m2, autoload_with=connection)
        else:
            with expect_warnings('Skipped unsupported reflection of expression-based index t_idx'):
                eq_(insp.get_indexes('t'), expected)
                m2 = MetaData()
                t2 = Table('t', m2, autoload_with=connection)
        self.compare_table_index_with_expected(t2, expected, connection.engine.name)

    @testing.requires.index_reflects_included_columns
    def test_reflect_covering_index(self, metadata, connection):
        t = Table('t', metadata, Column('x', String(30)), Column('y', String(30)))
        idx = Index('t_idx', t.c.x)
        idx.dialect_options[connection.engine.name]['include'] = ['y']
        metadata.create_all(connection)
        insp = inspect(connection)
        get_indexes = insp.get_indexes('t')
        eq_(get_indexes, [{'name': 't_idx', 'column_names': ['x'], 'include_columns': ['y'], 'unique': False, 'dialect_options': mock.ANY}])
        eq_(get_indexes[0]['dialect_options']['%s_include' % connection.engine.name], ['y'])
        t2 = Table('t', MetaData(), autoload_with=connection)
        eq_(list(t2.indexes)[0].dialect_options[connection.engine.name]['include'], ['y'])

    def _type_round_trip(self, connection, metadata, *types):
        t = Table('t', metadata, *[Column('t%d' % i, type_) for i, type_ in enumerate(types)])
        t.create(connection)
        return [c['type'] for c in inspect(connection).get_columns('t')]

    @testing.requires.table_reflection
    def test_numeric_reflection(self, connection, metadata):
        for typ in self._type_round_trip(connection, metadata, sql_types.Numeric(18, 5)):
            assert isinstance(typ, sql_types.Numeric)
            eq_(typ.precision, 18)
            eq_(typ.scale, 5)

    @testing.requires.table_reflection
    def test_varchar_reflection(self, connection, metadata):
        typ = self._type_round_trip(connection, metadata, sql_types.String(52))[0]
        assert isinstance(typ, sql_types.String)
        eq_(typ.length, 52)

    @testing.requires.table_reflection
    def test_nullable_reflection(self, connection, metadata):
        t = Table('t', metadata, Column('a', Integer, nullable=True), Column('b', Integer, nullable=False))
        t.create(connection)
        eq_({col['name']: col['nullable'] for col in inspect(connection).get_columns('t')}, {'a': True, 'b': False})

    @testing.combinations((None, 'CASCADE', None, testing.requires.foreign_key_constraint_option_reflection_ondelete), (None, None, 'SET NULL', testing.requires.foreign_key_constraint_option_reflection_onupdate), ({}, None, 'NO ACTION', testing.requires.foreign_key_constraint_option_reflection_onupdate), ({}, 'NO ACTION', None, testing.requires.fk_constraint_option_reflection_ondelete_noaction), (None, None, 'RESTRICT', testing.requires.fk_constraint_option_reflection_onupdate_restrict), (None, 'RESTRICT', None, testing.requires.fk_constraint_option_reflection_ondelete_restrict), argnames='expected,ondelete,onupdate')
    def test_get_foreign_key_options(self, connection, metadata, expected, ondelete, onupdate):
        options = {}
        if ondelete:
            options['ondelete'] = ondelete
        if onupdate:
            options['onupdate'] = onupdate
        if expected is None:
            expected = options
        Table('x', metadata, Column('id', Integer, primary_key=True), test_needs_fk=True)
        Table('table', metadata, Column('id', Integer, primary_key=True), Column('x_id', Integer, ForeignKey('x.id', name='xid')), Column('test', String(10)), test_needs_fk=True)
        Table('user', metadata, Column('id', Integer, primary_key=True), Column('name', String(50), nullable=False), Column('tid', Integer), sa.ForeignKeyConstraint(['tid'], ['table.id'], name='myfk', **options), test_needs_fk=True)
        metadata.create_all(connection)
        insp = inspect(connection)
        opts = insp.get_foreign_keys('table')[0]['options']
        eq_({k: opts[k] for k in opts if opts[k]}, {})
        opts = insp.get_foreign_keys('user')[0]['options']
        eq_(opts, expected)