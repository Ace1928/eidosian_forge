import errno
import http.client as http_client
import http.server as http_server
import os
import posixpath
import random
import re
import socket
import sys
from urllib.parse import urlparse
from .. import osutils, urlutils
from . import test_server
def _header_line_length(self, keyword, value):
    header_line = '{}: {}\r\n'.format(keyword, value)
    return len(header_line)