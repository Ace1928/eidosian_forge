from barbicanclient import client
from barbicanclient.tests import test_client
from barbicanclient.tests.utils import mock_delete_secret_for_responses
from barbicanclient.tests.utils import mock_get_secret_for_client
from barbicanclient.tests.v1.test_secrets import SecretData
from barbicanclient.v1 import secrets
from oslo_serialization import jsonutils
def _delete_from_manager_with_consumers(self, secret_ref, force=False):
    consumers = [{'service': 'service_test', 'resource_type': 'type_test', 'resource_id': 'id_test'}]
    self._delete_from_manager(secret_ref, force=force, consumers=consumers)