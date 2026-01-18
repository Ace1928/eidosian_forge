from barbicanclient import client
from barbicanclient.tests import test_client
from barbicanclient.tests.utils import mock_delete_secret_for_responses
from barbicanclient.tests.utils import mock_get_secret_for_client
from barbicanclient.tests.v1.test_secrets import SecretData
from barbicanclient.v1 import secrets
from oslo_serialization import jsonutils
def _register_consumer(self):
    data = self.secret.get_dict(self.entity_href, consumers=[self.secret.consumer])
    self.responses.post(self.entity_href + '/consumers/', json=data)
    return self.manager.register_consumer(self.entity_href, self.secret.consumer.get('service'), self.secret.consumer.get('resource_type'), self.secret.consumer.get('resource_id'))