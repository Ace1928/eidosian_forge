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
class CallbackRegistryFixture(fixtures.Fixture):
    """Callback registry fixture.

    This class is intended to be used as a fixture within unit tests and
    therefore consumers must register it using useFixture() within their
    unit test class. The implementation optionally allows consumers to pass
    in the CallbacksManager manager to use for your tests.
    """

    def __init__(self, callback_manager=None):
        """Creates a new RegistryFixture.

        :param callback_manager: If specified, the return value to use for
        _get_callback_manager(). Otherwise a new instance of CallbacksManager
        is used.
        """
        super().__init__()
        self.callback_manager = callback_manager or manager.CallbacksManager()
        self.patcher = None

    def _setUp(self):
        self._orig_manager = registry._get_callback_manager()
        self.patcher = mock.patch.object(registry, '_get_callback_manager', return_value=self.callback_manager)
        self.patcher.start()
        self.addCleanup(self._restore)

    def _restore(self):
        registry._CALLBACK_MANAGER = self._orig_manager
        self.patcher.stop()