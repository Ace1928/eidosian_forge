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
class QuotedNameArgumentTest(fixtures.TablesTest):
    run_create_tables = 'once'
    __backend__ = True

    @classmethod
    def define_tables(cls, metadata):
        Table("quote ' one", metadata, Column('id', Integer), Column('name', String(50)), Column('data', String(50)), Column('related_id', Integer), sa.PrimaryKeyConstraint('id', name="pk quote ' one"), sa.Index("ix quote ' one", 'name'), sa.UniqueConstraint('data', name="uq quote' one"), sa.ForeignKeyConstraint(['id'], ['related.id'], name="fk quote ' one"), sa.CheckConstraint("name != 'foo'", name="ck quote ' one"), comment="quote ' one comment", test_needs_fk=True)
        if testing.requires.symbol_names_w_double_quote.enabled:
            Table('quote " two', metadata, Column('id', Integer), Column('name', String(50)), Column('data', String(50)), Column('related_id', Integer), sa.PrimaryKeyConstraint('id', name='pk quote " two'), sa.Index('ix quote " two', 'name'), sa.UniqueConstraint('data', name='uq quote" two'), sa.ForeignKeyConstraint(['id'], ['related.id'], name='fk quote " two'), sa.CheckConstraint("name != 'foo'", name='ck quote " two '), comment='quote " two comment', test_needs_fk=True)
        Table('related', metadata, Column('id', Integer, primary_key=True), Column('related', Integer), test_needs_fk=True)
        if testing.requires.view_column_reflection.enabled:
            if testing.requires.symbol_names_w_double_quote.enabled:
                names = ["quote ' one", 'quote " two']
            else:
                names = ["quote ' one"]
            for name in names:
                query = 'CREATE VIEW %s AS SELECT * FROM %s' % (config.db.dialect.identifier_preparer.quote('view %s' % name), config.db.dialect.identifier_preparer.quote(name))
                event.listen(metadata, 'after_create', DDL(query))
                event.listen(metadata, 'before_drop', DDL('DROP VIEW %s' % config.db.dialect.identifier_preparer.quote('view %s' % name)))

    def quote_fixtures(fn):
        return testing.combinations(("quote ' one",), ('quote " two', testing.requires.symbol_names_w_double_quote))(fn)

    @quote_fixtures
    def test_get_table_options(self, name):
        insp = inspect(config.db)
        if testing.requires.reflect_table_options.enabled:
            res = insp.get_table_options(name)
            is_true(isinstance(res, dict))
        else:
            with expect_raises(NotImplementedError):
                res = insp.get_table_options(name)

    @quote_fixtures
    @testing.requires.view_column_reflection
    def test_get_view_definition(self, name):
        insp = inspect(config.db)
        assert insp.get_view_definition('view %s' % name)

    @quote_fixtures
    def test_get_columns(self, name):
        insp = inspect(config.db)
        assert insp.get_columns(name)

    @quote_fixtures
    def test_get_pk_constraint(self, name):
        insp = inspect(config.db)
        assert insp.get_pk_constraint(name)

    @quote_fixtures
    def test_get_foreign_keys(self, name):
        insp = inspect(config.db)
        assert insp.get_foreign_keys(name)

    @quote_fixtures
    def test_get_indexes(self, name):
        insp = inspect(config.db)
        assert insp.get_indexes(name)

    @quote_fixtures
    @testing.requires.unique_constraint_reflection
    def test_get_unique_constraints(self, name):
        insp = inspect(config.db)
        assert insp.get_unique_constraints(name)

    @quote_fixtures
    @testing.requires.comment_reflection
    def test_get_table_comment(self, name):
        insp = inspect(config.db)
        assert insp.get_table_comment(name)

    @quote_fixtures
    @testing.requires.check_constraint_reflection
    def test_get_check_constraints(self, name):
        insp = inspect(config.db)
        assert insp.get_check_constraints(name)