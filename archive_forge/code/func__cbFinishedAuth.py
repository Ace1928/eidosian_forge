import struct
from twisted.conch import error, interfaces
from twisted.conch.ssh import keys, service, transport
from twisted.conch.ssh.common import NS, getNS
from twisted.cred import credentials
from twisted.cred.error import UnauthorizedLogin
from twisted.internet import defer, reactor
from twisted.logger import Logger
from twisted.python import failure
from twisted.python.compat import nativeString
def _cbFinishedAuth(self, result):
    """
        The callback when user has successfully been authenticated.  For a
        description of the arguments, see L{twisted.cred.portal.Portal.login}.
        We start the service requested by the user.
        """
    interface, avatar, logout = result
    self.transport.avatar = avatar
    self.transport.logoutFunction = logout
    service = self.transport.factory.getService(self.transport, self.nextService)
    if not service:
        raise error.ConchError(f'could not get next service: {self.nextService}')
    self._log.debug('{user!r} authenticated with {method!r}', user=self.user, method=self.method)
    self.transport.sendPacket(MSG_USERAUTH_SUCCESS, b'')
    self.transport.setService(service())