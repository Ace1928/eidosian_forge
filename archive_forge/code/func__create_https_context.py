import email.parser
import email.message
import io
import re
from collections.abc import Iterable
from urllib.parse import urlsplit
from eventlet.green import http, os, socket
def _create_https_context(http_version):
    context = ssl._create_default_https_context()
    if http_version == 11:
        context.set_alpn_protocols(['http/1.1'])
    if context.post_handshake_auth is not None:
        context.post_handshake_auth = True
    return context