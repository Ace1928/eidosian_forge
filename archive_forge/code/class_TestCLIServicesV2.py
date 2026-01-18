from unittest import mock
from mistralclient.api.v2 import services
from mistralclient.commands.v2 import services as service_cmd
from mistralclient.tests.unit import base
class TestCLIServicesV2(base.BaseCommandTest):

    def test_list(self):
        self.client.services.list.return_value = [SERVICE]
        expected = (SERVICE_DICT['name'], SERVICE_DICT['type'])
        result = self.call(service_cmd.List)
        self.assertListEqual([expected], result[1])