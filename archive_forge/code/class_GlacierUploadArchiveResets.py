import json
import copy
import tempfile
from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.glacier.layer1 import Layer1
from boto.compat import six
class GlacierUploadArchiveResets(GlacierLayer1ConnectionBase):

    def test_upload_archive(self):
        fake_data = tempfile.NamedTemporaryFile()
        fake_data.write(b'foobarbaz')
        fake_data.seek(2)
        self.set_http_response(status_code=201)
        self.service_connection.connection.request.side_effect = lambda *args: fake_data.read()
        self.service_connection.upload_archive('vault_name', fake_data, 'linear_hash', 'tree_hash')
        self.assertEqual(fake_data.tell(), 2)