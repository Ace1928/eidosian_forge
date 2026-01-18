import importlib
from . import testing
from .. import assert_raises
from .. import config
from .. import engines
from .. import eq_
from .. import fixtures
from .. import is_not_none
from .. import is_true
from .. import ne_
from .. import provide_metadata
from ..assertions import expect_raises
from ..assertions import expect_raises_message
from ..config import requirements
from ..provision import set_default_schema_on_connection
from ..schema import Column
from ..schema import Table
from ... import bindparam
from ... import dialects
from ... import event
from ... import exc
from ... import Integer
from ... import literal_column
from ... import select
from ... import String
from ...sql.compiler import Compiled
from ...util import inspect_getfullargspec
class IsolationLevelTest(fixtures.TestBase):
    __backend__ = True
    __requires__ = ('isolation_level',)

    def _get_non_default_isolation_level(self):
        levels = requirements.get_isolation_levels(config)
        default = levels['default']
        supported = levels['supported']
        s = set(supported).difference(['AUTOCOMMIT', default])
        if s:
            return s.pop()
        else:
            config.skip_test('no non-default isolation level available')

    def test_default_isolation_level(self):
        eq_(config.db.dialect.default_isolation_level, requirements.get_isolation_levels(config)['default'])

    def test_non_default_isolation_level(self):
        non_default = self._get_non_default_isolation_level()
        with config.db.connect() as conn:
            existing = conn.get_isolation_level()
            ne_(existing, non_default)
            conn.execution_options(isolation_level=non_default)
            eq_(conn.get_isolation_level(), non_default)
            conn.dialect.reset_isolation_level(conn.connection.dbapi_connection)
            eq_(conn.get_isolation_level(), existing)

    def test_all_levels(self):
        levels = requirements.get_isolation_levels(config)
        all_levels = levels['supported']
        for level in set(all_levels).difference(['AUTOCOMMIT']):
            with config.db.connect() as conn:
                conn.execution_options(isolation_level=level)
                eq_(conn.get_isolation_level(), level)
                trans = conn.begin()
                trans.rollback()
                eq_(conn.get_isolation_level(), level)
            with config.db.connect() as conn:
                eq_(conn.get_isolation_level(), levels['default'])

    @testing.requires.get_isolation_level_values
    def test_invalid_level_execution_option(self, connection_no_trans):
        """test for the new get_isolation_level_values() method"""
        connection = connection_no_trans
        with expect_raises_message(exc.ArgumentError, "Invalid value '%s' for isolation_level. Valid isolation levels for '%s' are %s" % ('FOO', connection.dialect.name, ', '.join(requirements.get_isolation_levels(config)['supported']))):
            connection.execution_options(isolation_level='FOO')

    @testing.requires.get_isolation_level_values
    @testing.requires.dialect_level_isolation_level_param
    def test_invalid_level_engine_param(self, testing_engine):
        """test for the new get_isolation_level_values() method
        and support for the dialect-level 'isolation_level' parameter.

        """
        eng = testing_engine(options=dict(isolation_level='FOO'))
        with expect_raises_message(exc.ArgumentError, "Invalid value '%s' for isolation_level. Valid isolation levels for '%s' are %s" % ('FOO', eng.dialect.name, ', '.join(requirements.get_isolation_levels(config)['supported']))):
            eng.connect()

    @testing.requires.independent_readonly_connections
    def test_dialect_user_setting_is_restored(self, testing_engine):
        levels = requirements.get_isolation_levels(config)
        default = levels['default']
        supported = sorted(set(levels['supported']).difference([default, 'AUTOCOMMIT']))[0]
        e = testing_engine(options={'isolation_level': supported})
        with e.connect() as conn:
            eq_(conn.get_isolation_level(), supported)
        with e.connect() as conn:
            conn.execution_options(isolation_level=default)
            eq_(conn.get_isolation_level(), default)
        with e.connect() as conn:
            eq_(conn.get_isolation_level(), supported)