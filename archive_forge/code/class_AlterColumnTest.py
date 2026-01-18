from sqlalchemy import BigInteger
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import MetaData
from sqlalchemy import Table
from sqlalchemy.testing import in_
from ._autogen_fixtures import AutogenFixtureTest
from ... import testing
from ...testing import config
from ...testing import eq_
from ...testing import is_
from ...testing import TestBase
class AlterColumnTest(AutogenFixtureTest, TestBase):
    __backend__ = True

    @testing.combinations((True,), (False,))
    @config.requirements.comments
    def test_all_existings_filled(self, pk):
        m1 = MetaData()
        m2 = MetaData()
        Table('a', m1, Column('x', Integer, primary_key=pk))
        Table('a', m2, Column('x', Integer, comment='x', primary_key=pk))
        alter_col = self._assert_alter_col(m1, m2, pk)
        eq_(alter_col.modify_comment, 'x')

    @testing.combinations((True,), (False,))
    @config.requirements.comments
    def test_all_existings_filled_in_notnull(self, pk):
        m1 = MetaData()
        m2 = MetaData()
        Table('a', m1, Column('x', Integer, nullable=False, primary_key=pk))
        Table('a', m2, Column('x', Integer, nullable=False, comment='x', primary_key=pk))
        self._assert_alter_col(m1, m2, pk, nullable=False)

    @testing.combinations((True,), (False,))
    @config.requirements.comments
    def test_all_existings_filled_in_comment(self, pk):
        m1 = MetaData()
        m2 = MetaData()
        Table('a', m1, Column('x', Integer, comment='old', primary_key=pk))
        Table('a', m2, Column('x', Integer, comment='new', primary_key=pk))
        alter_col = self._assert_alter_col(m1, m2, pk)
        eq_(alter_col.existing_comment, 'old')

    @testing.combinations((True,), (False,))
    @config.requirements.comments
    def test_all_existings_filled_in_server_default(self, pk):
        m1 = MetaData()
        m2 = MetaData()
        Table('a', m1, Column('x', Integer, server_default='5', primary_key=pk))
        Table('a', m2, Column('x', Integer, server_default='5', comment='new', primary_key=pk))
        alter_col = self._assert_alter_col(m1, m2, pk)
        in_('5', alter_col.existing_server_default.arg.text)

    def _assert_alter_col(self, m1, m2, pk, nullable=None):
        ops = self._fixture(m1, m2, return_ops=True)
        modify_table = ops.ops[-1]
        alter_col = modify_table.ops[0]
        if nullable is None:
            eq_(alter_col.existing_nullable, not pk)
        else:
            eq_(alter_col.existing_nullable, nullable)
        assert alter_col.existing_type._compare_type_affinity(Integer())
        return alter_col