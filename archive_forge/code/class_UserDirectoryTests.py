from os.path import abspath
from xml.dom.minidom import parseString
from unittest import skipIf
from zope.interface.verify import verifyObject
from twisted.internet import defer, reactor
from twisted.logger import globalLogPublisher
from twisted.python import failure, filepath
from twisted.spread import pb
from twisted.spread.banana import SIZE_LIMIT
from twisted.test import proto_helpers
from twisted.trial.unittest import TestCase
from twisted.web import client, distrib, resource, server, static
from twisted.web.http_headers import Headers
from twisted.web.test._util import _render
from twisted.web.test.requesthelper import DummyChannel, DummyRequest
class UserDirectoryTests(TestCase):
    """
    Tests for L{UserDirectory}, a resource for listing all user resources
    available on a system.
    """

    def setUp(self):
        self.alice = ('alice', 'x', 123, 456, 'Alice,,,', self.mktemp(), '/bin/sh')
        self.bob = ('bob', 'x', 234, 567, 'Bob,,,', self.mktemp(), '/bin/sh')
        self.database = _PasswordDatabase([self.alice, self.bob])
        self.directory = distrib.UserDirectory(self.database)

    def test_interface(self):
        """
        L{UserDirectory} instances provide L{resource.IResource}.
        """
        self.assertTrue(verifyObject(resource.IResource, self.directory))

    async def _404Test(self, name: bytes) -> None:
        """
        Verify that requesting the C{name} child of C{self.directory} results
        in a 404 response.
        """
        request = DummyRequest([name])
        result = self.directory.getChild(name, request)
        d = _render(result, request)
        await d
        self.assertEqual(request.responseCode, 404)

    async def test_getInvalidUser(self):
        """
        L{UserDirectory.getChild} returns a resource which renders a 404
        response when passed a string which does not correspond to any known
        user.
        """
        await self._404Test(b'carol')

    async def test_getUserWithoutResource(self):
        """
        L{UserDirectory.getChild} returns a resource which renders a 404
        response when passed a string which corresponds to a known user who has
        neither a user directory nor a user distrib socket.
        """
        await self._404Test(b'alice')

    def test_getPublicHTMLChild(self):
        """
        L{UserDirectory.getChild} returns a L{static.File} instance when passed
        the name of a user with a home directory containing a I{public_html}
        directory.
        """
        home = filepath.FilePath(self.bob[-2])
        public_html = home.child('public_html')
        public_html.makedirs()
        request = DummyRequest(['bob'])
        result = self.directory.getChild(b'bob', request)
        self.assertIsInstance(result, static.File)
        self.assertEqual(result.path, public_html.path)

    def test_getDistribChild(self):
        """
        L{UserDirectory.getChild} returns a L{ResourceSubscription} instance
        when passed the name of a user suffixed with C{".twistd"} who has a
        home directory containing a I{.twistd-web-pb} socket.
        """
        home = filepath.FilePath(self.bob[-2])
        home.makedirs()
        web = home.child('.twistd-web-pb')
        request = DummyRequest(['bob'])
        result = self.directory.getChild(b'bob.twistd', request)
        self.assertIsInstance(result, distrib.ResourceSubscription)
        self.assertEqual(result.host, 'unix')
        self.assertEqual(abspath(result.port), web.path)

    def test_invalidMethod(self):
        """
        L{UserDirectory.render} raises L{UnsupportedMethod} in response to a
        non-I{GET} request.
        """
        request = DummyRequest([''])
        request.method = 'POST'
        self.assertRaises(server.UnsupportedMethod, self.directory.render, request)

    def test_render(self):
        """
        L{UserDirectory} renders a list of links to available user content
        in response to a I{GET} request.
        """
        public_html = filepath.FilePath(self.alice[-2]).child('public_html')
        public_html.makedirs()
        web = filepath.FilePath(self.bob[-2])
        web.makedirs()
        web.child('.twistd-web-pb').setContent(b'')
        request = DummyRequest([''])
        result = _render(self.directory, request)

        def cbRendered(ignored):
            document = parseString(b''.join(request.written))
            [alice, bob] = document.getElementsByTagName('li')
            self.assertEqual(alice.firstChild.tagName, 'a')
            self.assertEqual(alice.firstChild.getAttribute('href'), 'alice/')
            self.assertEqual(alice.firstChild.firstChild.data, 'Alice (file)')
            self.assertEqual(bob.firstChild.tagName, 'a')
            self.assertEqual(bob.firstChild.getAttribute('href'), 'bob.twistd/')
            self.assertEqual(bob.firstChild.firstChild.data, 'Bob (twistd)')
        result.addCallback(cbRendered)
        return result

    @skipIf(not pwd, 'pwd module required')
    def test_passwordDatabase(self):
        """
        If L{UserDirectory} is instantiated with no arguments, it uses the
        L{pwd} module as its password database.
        """
        directory = distrib.UserDirectory()
        self.assertIdentical(directory._pwd, pwd)