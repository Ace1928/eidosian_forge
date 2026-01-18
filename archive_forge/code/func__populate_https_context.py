import email.parser
import email.message
import io
import re
from collections.abc import Iterable
from urllib.parse import urlsplit
from eventlet.green import http, os, socket
def _populate_https_context(context, check_hostname):
    if check_hostname is not None:
        context.check_hostname = check_hostname