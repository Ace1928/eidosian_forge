import email.parser
import email.message
import errno
import http
import io
import re
import socket
import sys
import collections.abc
from urllib.parse import urlsplit
def _close_conn(self):
    fp = self.fp
    self.fp = None
    fp.close()