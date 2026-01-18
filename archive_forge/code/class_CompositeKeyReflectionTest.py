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
class CompositeKeyReflectionTest(fixtures.TablesTest):
    __backend__ = True

    @classmethod
    def define_tables(cls, metadata):
        tb1 = Table('tb1', metadata, Column('id', Integer), Column('attr', Integer), Column('name', sql_types.VARCHAR(20)), sa.PrimaryKeyConstraint('name', 'id', 'attr', name='pk_tb1'), schema=None, test_needs_fk=True)
        Table('tb2', metadata, Column('id', Integer, primary_key=True), Column('pid', Integer), Column('pattr', Integer), Column('pname', sql_types.VARCHAR(20)), sa.ForeignKeyConstraint(['pname', 'pid', 'pattr'], [tb1.c.name, tb1.c.id, tb1.c.attr], name='fk_tb1_name_id_attr'), schema=None, test_needs_fk=True)

    @testing.requires.primary_key_constraint_reflection
    def test_pk_column_order(self, connection):
        insp = inspect(connection)
        primary_key = insp.get_pk_constraint(self.tables.tb1.name)
        eq_(primary_key.get('constrained_columns'), ['name', 'id', 'attr'])

    @testing.requires.foreign_key_constraint_reflection
    def test_fk_column_order(self, connection):
        insp = inspect(connection)
        foreign_keys = insp.get_foreign_keys(self.tables.tb2.name)
        eq_(len(foreign_keys), 1)
        fkey1 = foreign_keys[0]
        eq_(fkey1.get('referred_columns'), ['name', 'id', 'attr'])
        eq_(fkey1.get('constrained_columns'), ['pname', 'pid', 'pattr'])