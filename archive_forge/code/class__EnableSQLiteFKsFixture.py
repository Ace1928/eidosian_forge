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
class _EnableSQLiteFKsFixture(fixtures.Fixture):
    """Turn SQLite PRAGMA foreign keys on and off for tests.

    FIXME(zzzeek): figure out some way to get oslo.db test_base to honor
    oslo_db.engines.create_engine() arguments like sqlite_fks as well
    as handling that it needs to be turned off during drops.

    """

    def __init__(self, engine):
        self.engine = engine

    def _setUp(self):
        if self.engine.name == 'sqlite':
            with self.engine.connect() as conn:
                cursor = conn.connection.cursor()
                cursor.execute('PRAGMA foreign_keys=ON')
                cursor.close()