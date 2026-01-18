import requests
from novaclient import api_versions
from novaclient import base
from novaclient import exceptions
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import flavors
class BytesWithMetaTest(utils.TestCase):

    def test_bytes_with_meta(self):
        resp = create_response_obj_with_header()
        obj = base.BytesWithMeta(b'test-bytes', resp)
        self.assertEqual(b'test-bytes', obj)
        self.assertTrue(hasattr(obj, 'request_ids'))
        self.assertEqual(fakes.FAKE_REQUEST_ID_LIST, obj.request_ids)