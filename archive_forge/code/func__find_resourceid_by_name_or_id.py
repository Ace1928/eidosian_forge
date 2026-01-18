from unittest import mock
import uuid
from designateclient import exceptions
from designateclient.tests import base
from designateclient import utils
def _find_resourceid_by_name_or_id(self, name_or_id, by_name=False):
    resource_client = mock.Mock()
    resource_client.list.return_value = LIST_MOCK_RESPONSE
    resourceid = utils.find_resourceid_by_name_or_id(resource_client, name_or_id)
    self.assertEqual(by_name, resource_client.list.called)
    return resourceid