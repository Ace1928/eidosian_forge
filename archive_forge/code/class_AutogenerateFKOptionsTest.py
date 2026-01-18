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
class AutogenerateFKOptionsTest(AutogenFixtureTest, TestBase):
    __backend__ = True

    def _fk_opts_fixture(self, old_opts, new_opts):
        m1 = MetaData()
        m2 = MetaData()
        Table('some_table', m1, Column('id', Integer, primary_key=True), Column('test', String(10)))
        Table('user', m1, Column('id', Integer, primary_key=True), Column('name', String(50), nullable=False), Column('tid', Integer), ForeignKeyConstraint(['tid'], ['some_table.id'], **old_opts))
        Table('some_table', m2, Column('id', Integer, primary_key=True), Column('test', String(10)))
        Table('user', m2, Column('id', Integer, primary_key=True), Column('name', String(50), nullable=False), Column('tid', Integer), ForeignKeyConstraint(['tid'], ['some_table.id'], **new_opts))
        return self._fixture(m1, m2)

    @config.requirements.fk_ondelete_is_reflected
    def test_add_ondelete(self):
        diffs = self._fk_opts_fixture({}, {'ondelete': 'cascade'})
        self._assert_fk_diff(diffs[0], 'remove_fk', 'user', ['tid'], 'some_table', ['id'], ondelete=None, conditional_name='servergenerated')
        self._assert_fk_diff(diffs[1], 'add_fk', 'user', ['tid'], 'some_table', ['id'], ondelete='cascade')

    @config.requirements.fk_ondelete_is_reflected
    def test_remove_ondelete(self):
        diffs = self._fk_opts_fixture({'ondelete': 'CASCADE'}, {})
        self._assert_fk_diff(diffs[0], 'remove_fk', 'user', ['tid'], 'some_table', ['id'], ondelete='CASCADE', conditional_name='servergenerated')
        self._assert_fk_diff(diffs[1], 'add_fk', 'user', ['tid'], 'some_table', ['id'], ondelete=None)

    def test_nochange_ondelete(self):
        """test case sensitivity"""
        diffs = self._fk_opts_fixture({'ondelete': 'caSCAde'}, {'ondelete': 'CasCade'})
        eq_(diffs, [])

    @config.requirements.fk_onupdate_is_reflected
    def test_add_onupdate(self):
        diffs = self._fk_opts_fixture({}, {'onupdate': 'cascade'})
        self._assert_fk_diff(diffs[0], 'remove_fk', 'user', ['tid'], 'some_table', ['id'], onupdate=None, conditional_name='servergenerated')
        self._assert_fk_diff(diffs[1], 'add_fk', 'user', ['tid'], 'some_table', ['id'], onupdate='cascade')

    @config.requirements.fk_onupdate_is_reflected
    def test_remove_onupdate(self):
        diffs = self._fk_opts_fixture({'onupdate': 'CASCADE'}, {})
        self._assert_fk_diff(diffs[0], 'remove_fk', 'user', ['tid'], 'some_table', ['id'], onupdate='CASCADE', conditional_name='servergenerated')
        self._assert_fk_diff(diffs[1], 'add_fk', 'user', ['tid'], 'some_table', ['id'], onupdate=None)

    @config.requirements.fk_onupdate
    def test_nochange_onupdate(self):
        """test case sensitivity"""
        diffs = self._fk_opts_fixture({'onupdate': 'caSCAde'}, {'onupdate': 'CasCade'})
        eq_(diffs, [])

    @config.requirements.fk_ondelete_restrict
    def test_nochange_ondelete_restrict(self):
        """test the RESTRICT option which MySQL doesn't report on"""
        diffs = self._fk_opts_fixture({'ondelete': 'restrict'}, {'ondelete': 'restrict'})
        eq_(diffs, [])

    @config.requirements.fk_onupdate_restrict
    def test_nochange_onupdate_restrict(self):
        """test the RESTRICT option which MySQL doesn't report on"""
        diffs = self._fk_opts_fixture({'onupdate': 'restrict'}, {'onupdate': 'restrict'})
        eq_(diffs, [])

    @config.requirements.fk_ondelete_noaction
    def test_nochange_ondelete_noaction(self):
        """test the NO ACTION option which generally comes back as None"""
        diffs = self._fk_opts_fixture({'ondelete': 'no action'}, {'ondelete': 'no action'})
        eq_(diffs, [])

    @config.requirements.fk_onupdate
    def test_nochange_onupdate_noaction(self):
        """test the NO ACTION option which generally comes back as None"""
        diffs = self._fk_opts_fixture({'onupdate': 'no action'}, {'onupdate': 'no action'})
        eq_(diffs, [])

    @config.requirements.fk_ondelete_restrict
    def test_change_ondelete_from_restrict(self):
        """test the RESTRICT option which MySQL doesn't report on"""
        diffs = self._fk_opts_fixture({'ondelete': 'restrict'}, {'ondelete': 'cascade'})
        self._assert_fk_diff(diffs[0], 'remove_fk', 'user', ['tid'], 'some_table', ['id'], onupdate=None, ondelete=mock.ANY, conditional_name='servergenerated')
        self._assert_fk_diff(diffs[1], 'add_fk', 'user', ['tid'], 'some_table', ['id'], onupdate=None, ondelete='cascade')

    @config.requirements.fk_ondelete_restrict
    def test_change_onupdate_from_restrict(self):
        """test the RESTRICT option which MySQL doesn't report on"""
        diffs = self._fk_opts_fixture({'onupdate': 'restrict'}, {'onupdate': 'cascade'})
        self._assert_fk_diff(diffs[0], 'remove_fk', 'user', ['tid'], 'some_table', ['id'], onupdate=mock.ANY, ondelete=None, conditional_name='servergenerated')
        self._assert_fk_diff(diffs[1], 'add_fk', 'user', ['tid'], 'some_table', ['id'], onupdate='cascade', ondelete=None)

    @config.requirements.fk_ondelete_is_reflected
    @config.requirements.fk_onupdate_is_reflected
    def test_ondelete_onupdate_combo(self):
        diffs = self._fk_opts_fixture({'onupdate': 'CASCADE', 'ondelete': 'SET NULL'}, {'onupdate': 'RESTRICT', 'ondelete': 'RESTRICT'})
        self._assert_fk_diff(diffs[0], 'remove_fk', 'user', ['tid'], 'some_table', ['id'], onupdate='CASCADE', ondelete='SET NULL', conditional_name='servergenerated')
        self._assert_fk_diff(diffs[1], 'add_fk', 'user', ['tid'], 'some_table', ['id'], onupdate='RESTRICT', ondelete='RESTRICT')

    @config.requirements.fk_initially
    def test_add_initially_deferred(self):
        diffs = self._fk_opts_fixture({}, {'initially': 'deferred'})
        self._assert_fk_diff(diffs[0], 'remove_fk', 'user', ['tid'], 'some_table', ['id'], initially=None, conditional_name='servergenerated')
        self._assert_fk_diff(diffs[1], 'add_fk', 'user', ['tid'], 'some_table', ['id'], initially='deferred')

    @config.requirements.fk_initially
    def test_remove_initially_deferred(self):
        diffs = self._fk_opts_fixture({'initially': 'deferred'}, {})
        self._assert_fk_diff(diffs[0], 'remove_fk', 'user', ['tid'], 'some_table', ['id'], initially='DEFERRED', deferrable=True, conditional_name='servergenerated')
        self._assert_fk_diff(diffs[1], 'add_fk', 'user', ['tid'], 'some_table', ['id'], initially=None)

    @config.requirements.fk_deferrable
    @config.requirements.fk_initially
    def test_add_initially_immediate_plus_deferrable(self):
        diffs = self._fk_opts_fixture({}, {'initially': 'immediate', 'deferrable': True})
        self._assert_fk_diff(diffs[0], 'remove_fk', 'user', ['tid'], 'some_table', ['id'], initially=None, conditional_name='servergenerated')
        self._assert_fk_diff(diffs[1], 'add_fk', 'user', ['tid'], 'some_table', ['id'], initially='immediate', deferrable=True)

    @config.requirements.fk_deferrable
    @config.requirements.fk_initially
    def test_remove_initially_immediate_plus_deferrable(self):
        diffs = self._fk_opts_fixture({'initially': 'immediate', 'deferrable': True}, {})
        self._assert_fk_diff(diffs[0], 'remove_fk', 'user', ['tid'], 'some_table', ['id'], initially=None, deferrable=True, conditional_name='servergenerated')
        self._assert_fk_diff(diffs[1], 'add_fk', 'user', ['tid'], 'some_table', ['id'], initially=None, deferrable=None)

    @config.requirements.fk_initially
    @config.requirements.fk_deferrable
    def test_add_initially_deferrable_nochange_one(self):
        diffs = self._fk_opts_fixture({'deferrable': True, 'initially': 'immediate'}, {'deferrable': True, 'initially': 'immediate'})
        eq_(diffs, [])

    @config.requirements.fk_initially
    @config.requirements.fk_deferrable
    def test_add_initially_deferrable_nochange_two(self):
        diffs = self._fk_opts_fixture({'deferrable': True, 'initially': 'deferred'}, {'deferrable': True, 'initially': 'deferred'})
        eq_(diffs, [])

    @config.requirements.fk_initially
    @config.requirements.fk_deferrable
    def test_add_initially_deferrable_nochange_three(self):
        diffs = self._fk_opts_fixture({'deferrable': None, 'initially': 'deferred'}, {'deferrable': None, 'initially': 'deferred'})
        eq_(diffs, [])

    @config.requirements.fk_deferrable
    def test_add_deferrable(self):
        diffs = self._fk_opts_fixture({}, {'deferrable': True})
        self._assert_fk_diff(diffs[0], 'remove_fk', 'user', ['tid'], 'some_table', ['id'], deferrable=None, conditional_name='servergenerated')
        self._assert_fk_diff(diffs[1], 'add_fk', 'user', ['tid'], 'some_table', ['id'], deferrable=True)

    @config.requirements.fk_deferrable_is_reflected
    def test_remove_deferrable(self):
        diffs = self._fk_opts_fixture({'deferrable': True}, {})
        self._assert_fk_diff(diffs[0], 'remove_fk', 'user', ['tid'], 'some_table', ['id'], deferrable=True, conditional_name='servergenerated')
        self._assert_fk_diff(diffs[1], 'add_fk', 'user', ['tid'], 'some_table', ['id'], deferrable=None)