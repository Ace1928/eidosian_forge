import base64
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from barbicanclient import base
from barbicanclient import exceptions
from barbicanclient.tests import test_client
from barbicanclient.tests.utils import mock_get_secret_for_client
from barbicanclient.v1 import acls
from barbicanclient.v1 import secrets
class SecretData(object):

    def __init__(self):
        self.name = 'Self destruction sequence'
        self.payload = 'the magic words are squeamish ossifrage'
        self.payload_content_type = 'text/plain'
        self.algorithm = 'AES'
        self.created = str(timeutils.utcnow())
        self.consumer = {'service': 'service_test', 'resource_type': 'type_test', 'resource_id': 'id_test'}
        self.secret_dict = {'name': self.name, 'status': 'ACTIVE', 'algorithm': self.algorithm, 'created': self.created}

    def get_dict(self, secret_ref=None, content_types_dict=None, consumers=None):
        secret = self.secret_dict
        if secret_ref:
            secret['secret_ref'] = secret_ref
        if content_types_dict:
            secret['content_types'] = content_types_dict
        if consumers:
            secret['consumers'] = consumers
        return secret