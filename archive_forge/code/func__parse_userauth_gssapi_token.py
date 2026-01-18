import weakref
import threading
import time
import re
from paramiko.common import (
from paramiko.message import Message
from paramiko.util import b, u
from paramiko.ssh_exception import (
from paramiko.server import InteractiveQuery
from paramiko.ssh_gss import GSSAuth, GSS_EXCEPTIONS
def _parse_userauth_gssapi_token(self, m):
    client_token = m.get_string()
    sshgss = self.sshgss
    try:
        token = sshgss.ssh_accept_sec_context(self.gss_host, client_token, self.auth_username)
    except Exception as e:
        self.transport.saved_exception = e
        result = AUTH_FAILED
        self._restore_delegate_auth_handler()
        self._send_auth_result(self.auth_username, self.method, result)
        raise
    if token is not None:
        m = Message()
        m.add_byte(cMSG_USERAUTH_GSSAPI_TOKEN)
        m.add_string(token)
        self.transport._expected_packet = (MSG_USERAUTH_GSSAPI_TOKEN, MSG_USERAUTH_GSSAPI_MIC, MSG_USERAUTH_REQUEST)
        self.transport._send_message(m)