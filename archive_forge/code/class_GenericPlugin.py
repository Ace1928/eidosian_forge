import abc
import collections
import urllib
import uuid
from keystoneauth1 import _utils
from keystoneauth1 import access
from keystoneauth1 import adapter
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1 import identity
from keystoneauth1 import plugin
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
class GenericPlugin(plugin.BaseAuthPlugin):
    BAD_TOKEN = uuid.uuid4().hex

    def __init__(self):
        super(GenericPlugin, self).__init__()
        self.endpoint = 'http://keystone.host:5000'
        self.headers = {'headerA': 'valueA', 'headerB': 'valueB'}
        self.cert = '/path/to/cert'
        self.connection_params = {'cert': self.cert, 'verify': False}

    def url(self, prefix):
        return '%s/%s' % (self.endpoint, prefix)

    def get_token(self, session, **kwargs):
        return self.BAD_TOKEN

    def get_headers(self, session, **kwargs):
        return self.headers

    def get_endpoint(self, session, **kwargs):
        return self.endpoint

    def get_connection_params(self, session, **kwargs):
        return self.connection_params