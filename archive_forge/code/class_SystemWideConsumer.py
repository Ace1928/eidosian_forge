import os
import platform
import socket
import stat
import six
from oauthlib import oauth1
from six.moves.urllib.parse import parse_qs, urlencode
from lazr.restfulclient.authorize import HttpAuthorizer
from lazr.restfulclient.errors import CredentialsFileError
class SystemWideConsumer(Consumer):
    """A consumer associated with the logged-in user rather than an app.

    This can be used to share a single OAuth token among multiple
    desktop applications. The OAuth consumer key will be derived from
    system information (platform and hostname).
    """
    KEY_FORMAT = 'System-wide: %s (%s)'

    def __init__(self, application_name, secret=''):
        """Constructor.

        :param application_name: An application name. This will be
            used in the User-Agent header.
        :param secret: The OAuth consumer secret. Don't use this. It's
            a misfeature, and lazr.restful doesn't expect it.
        """
        super(SystemWideConsumer, self).__init__(self.consumer_key, secret, application_name)

    @property
    def consumer_key(self):
        """The system-wide OAuth consumer key for this computer.

        This key identifies the platform and the computer's
        hostname. It does not identify the active user.
        """
        try:
            import distro
            distname = distro.name()
        except Exception:
            distname = ''
        if distname == '':
            distname = platform.system()
        return self.KEY_FORMAT % (distname, socket.gethostname())