import copy
from unittest import mock
import warnings
import fixtures
from oslo_config import cfg
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import provision
from oslo_db.sqlalchemy import session
from oslo_messaging import conffixture
from neutron_lib.api import attributes
from neutron_lib.api import definitions
from neutron_lib.callbacks import manager
from neutron_lib.callbacks import registry
from neutron_lib.db import api as db_api
from neutron_lib.db import model_base
from neutron_lib.db import model_query
from neutron_lib.db import resource_extend
from neutron_lib.plugins import directory
from neutron_lib import rpc
from neutron_lib.tests.unit import fake_notifier
class SqlFixture(fixtures.Fixture):

    @classmethod
    def _generate_schema(cls, engine):
        model_base.BASEV2.metadata.create_all(engine)

    def _delete_from_schema(self, engine):
        with engine.begin() as conn:
            for table in reversed(model_base.BASEV2.metadata.sorted_tables):
                conn.execute(table.delete())

    def _init_resources(self):
        pass

    def _setUp(self):
        self._init_resources()
        if not hasattr(self, 'engine'):
            return
        engine = self.engine
        self.addCleanup(lambda: self._delete_from_schema(engine))
        self.sessionmaker = session.get_maker(engine)
        _restore_factory = db_api.get_context_manager()._root_factory
        self.enginefacade_factory = enginefacade._TestTransactionFactory(self.engine, self.sessionmaker, from_factory=_restore_factory, apply_global=False)
        db_api.get_context_manager()._root_factory = self.enginefacade_factory
        engine = db_api.CONTEXT_WRITER.get_engine()
        self.addCleanup(lambda: setattr(db_api.get_context_manager(), '_root_factory', _restore_factory))
        self.useFixture(_EnableSQLiteFKsFixture(engine))