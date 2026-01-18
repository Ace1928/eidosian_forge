from unittest import mock
from designateclient import client as designate_client
from heat.common import exception as heat_exception
from heat.engine.clients.os import designate as client
from heat.tests import common
class DesignateZoneConstraintTest(common.HeatTestCase):

    def test_expected_exceptions(self):
        self.assertEqual((heat_exception.EntityNotFound,), client.DesignateZoneConstraint.expected_exceptions, 'DesignateZoneConstraint expected exceptions error')

    def test_constrain(self):
        constrain = client.DesignateZoneConstraint()
        client_mock = mock.MagicMock()
        client_plugin_mock = mock.MagicMock()
        client_plugin_mock.get_zone_id.return_value = None
        client_mock.client_plugin.return_value = client_plugin_mock
        self.assertIsNone(constrain.validate_with_client(client_mock, 'zone_1'))
        client_plugin_mock.get_zone_id.assert_called_once_with('zone_1')