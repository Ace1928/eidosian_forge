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
class PlacementAPIClientFixture(fixtures.Fixture):
    """Placement API client fixture.

    This class is intended to be used as a fixture within unit tests and
    therefore consumers must register it using useFixture() within their
    unit test class.
    """

    def __init__(self, placement_api_client):
        """Creates a new PlacementAPIClientFixture.

        :param placement_api_client: Placement API client object.
        """
        super().__init__()
        self.placement_api_client = placement_api_client

    def _setUp(self):
        self.addCleanup(self._restore)

        def mock_create_client():
            self.placement_api_client.client = mock.Mock()
        self._mock_create_client = mock.patch.object(self.placement_api_client, '_create_client', side_effect=mock_create_client)
        self._mock_get = mock.patch.object(self.placement_api_client, '_get')
        self._mock_post = mock.patch.object(self.placement_api_client, '_post')
        self._mock_put = mock.patch.object(self.placement_api_client, '_put')
        self._mock_delete = mock.patch.object(self.placement_api_client, '_delete')
        self._mock_create_client.start()
        self.mock_get = self._mock_get.start()
        self.mock_post = self._mock_post.start()
        self.mock_put = self._mock_put.start()
        self.mock_delete = self._mock_delete.start()

    def _restore(self):
        self._mock_create_client.stop()
        self._mock_get.stop()
        self._mock_post.stop()
        self._mock_put.stop()
        self._mock_delete.stop()