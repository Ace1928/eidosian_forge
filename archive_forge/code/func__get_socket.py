import socket
import io
import re
import email.utils
import email.message
import email.generator
import base64
import hmac
import copy
import datetime
import sys
from email.base64mime import body_encode as encode_base64
def _get_socket(self, host, port, timeout):
    if self.debuglevel > 0:
        self._print_debug('connect:', (host, port))
    new_socket = super()._get_socket(host, port, timeout)
    new_socket = self.context.wrap_socket(new_socket, server_hostname=self._host)
    return new_socket