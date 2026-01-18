import errno
import io
import json
import testtools
from urllib import parse
from glanceclient.tests import utils
from glanceclient.v1 import client
from glanceclient.v1 import images
from glanceclient.v1 import shell
class UrlParameterTest(testtools.TestCase):

    def setUp(self):
        super(UrlParameterTest, self).setUp()
        self.api = ParameterFakeAPI({})
        self.gc = client.Client('http://fakeaddress.com')
        self.gc.images = images.ImageManager(self.api)

    def test_is_public_list(self):
        shell.do_image_list(self.gc, FakeArg({'is_public': 'True'}))
        parts = parse.urlparse(self.api.url)
        qs_dict = parse.parse_qs(parts.query)
        self.assertIn('is_public', qs_dict)
        self.assertTrue(qs_dict['is_public'][0].lower() == 'true')