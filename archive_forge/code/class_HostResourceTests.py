from twisted.internet.defer import gatherResults
from twisted.trial.unittest import TestCase
from twisted.web.http import NOT_FOUND
from twisted.web.resource import NoResource
from twisted.web.server import Site
from twisted.web.static import Data
from twisted.web.test._util import _render
from twisted.web.test.test_web import DummyRequest
from twisted.web.vhost import NameVirtualHost, VHostMonsterResource, _HostResource
class HostResourceTests(TestCase):
    """
    Tests for L{_HostResource}.
    """

    def test_getChild(self):
        """
        L{_HostResource.getChild} returns the proper I{Resource} for the vhost
        embedded in the URL.  Verify that returning the proper I{Resource}
        required changing the I{Host} in the header.
        """
        bazroot = Data(b'root data', '')
        bazuri = Data(b'uri data', '')
        baztest = Data(b'test data', '')
        bazuri.putChild(b'test', baztest)
        bazroot.putChild(b'uri', bazuri)
        hr = _HostResource()
        root = NameVirtualHost()
        root.default = Data(b'default data', '')
        root.addHost(b'baz.com', bazroot)
        request = DummyRequest([b'uri', b'test'])
        request.prepath = [b'bar', b'http', b'baz.com']
        request.site = Site(root)
        request.isSecure = lambda: False
        request.host = b''
        step = hr.getChild(b'baz.com', request)
        self.assertIsInstance(step, Data)
        request = DummyRequest([b'uri', b'test'])
        step = root.getChild(b'uri', request)
        self.assertIsInstance(step, NoResource)