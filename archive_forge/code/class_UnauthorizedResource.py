from zope.interface import implementer
from twisted.cred import error
from twisted.cred.credentials import Anonymous
from twisted.logger import Logger
from twisted.python.components import proxyForInterface
from twisted.web import util
from twisted.web.resource import IResource, _UnsafeErrorPage
@implementer(IResource)
class UnauthorizedResource:
    """
    Simple IResource to escape Resource dispatch
    """
    isLeaf = True

    def __init__(self, factories):
        self._credentialFactories = factories

    def render(self, request):
        """
        Send www-authenticate headers to the client
        """

        def ensureBytes(s):
            return s.encode('ascii') if isinstance(s, str) else s

        def generateWWWAuthenticate(scheme, challenge):
            lst = []
            for k, v in challenge.items():
                k = ensureBytes(k)
                v = ensureBytes(v)
                lst.append(k + b'=' + quoteString(v))
            return b' '.join([scheme, b', '.join(lst)])

        def quoteString(s):
            return b'"' + s.replace(b'\\', b'\\\\').replace(b'"', b'\\"') + b'"'
        request.setResponseCode(401)
        for fact in self._credentialFactories:
            challenge = fact.getChallenge(request)
            request.responseHeaders.addRawHeader(b'www-authenticate', generateWWWAuthenticate(fact.scheme, challenge))
        if request.method == b'HEAD':
            return b''
        return b'Unauthorized'

    def getChildWithDefault(self, path, request):
        """
        Disable resource dispatch
        """
        return self

    def putChild(self, path, child):
        raise NotImplementedError()