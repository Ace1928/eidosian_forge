import errno
import smtplib
import socket
from email.utils import getaddresses, parseaddr
from . import config, osutils
from .errors import BzrError, InternalBzrError
def _authenticate(self):
    """If necessary authenticate yourself to the server."""
    auth = config.AuthenticationConfig()
    if self._smtp_username is None:
        self._smtp_username = auth.get_user('smtp', self._smtp_server)
        if self._smtp_username is None:
            return
    if self._smtp_password is None:
        self._smtp_password = auth.get_password('smtp', self._smtp_server, self._smtp_username)
    username = osutils.safe_utf8(self._smtp_username)
    password = osutils.safe_utf8(self._smtp_password)
    self._connection.login(username, password)