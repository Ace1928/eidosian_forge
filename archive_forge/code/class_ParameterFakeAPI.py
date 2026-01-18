import errno
import io
import json
import testtools
from urllib import parse
from glanceclient.tests import utils
from glanceclient.v1 import client
from glanceclient.v1 import images
from glanceclient.v1 import shell
class ParameterFakeAPI(utils.FakeAPI):
    image_list = {'images': [{'id': 'a', 'name': 'image-1', 'properties': {'arch': 'x86_64'}}, {'id': 'b', 'name': 'image-2', 'properties': {'arch': 'x86_64'}}]}

    def get(self, url, **kwargs):
        self.url = url
        return (utils.FakeResponse({}), ParameterFakeAPI.image_list)