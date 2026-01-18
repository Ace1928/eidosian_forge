from sqlalchemy import Column
from sqlalchemy import ForeignKeyConstraint
from sqlalchemy import Integer
from sqlalchemy import MetaData
from sqlalchemy import String
from sqlalchemy import Table
from ._autogen_fixtures import AutogenFixtureTest
from ...testing import combinations
from ...testing import config
from ...testing import eq_
from ...testing import mock
from ...testing import TestBase
def _fk_opts_fixture(self, old_opts, new_opts):
    m1 = MetaData()
    m2 = MetaData()
    Table('some_table', m1, Column('id', Integer, primary_key=True), Column('test', String(10)))
    Table('user', m1, Column('id', Integer, primary_key=True), Column('name', String(50), nullable=False), Column('tid', Integer), ForeignKeyConstraint(['tid'], ['some_table.id'], **old_opts))
    Table('some_table', m2, Column('id', Integer, primary_key=True), Column('test', String(10)))
    Table('user', m2, Column('id', Integer, primary_key=True), Column('name', String(50), nullable=False), Column('tid', Integer), ForeignKeyConstraint(['tid'], ['some_table.id'], **new_opts))
    return self._fixture(m1, m2)