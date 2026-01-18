import sys
import socket
from socket import _GLOBAL_DEFAULT_TIMEOUT
def acct(self, password):
    """Send new account name."""
    cmd = 'ACCT ' + password
    return self.voidcmd(cmd)