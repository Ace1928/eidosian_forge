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
class LineTooLong(HTTPException):

    def __init__(self, line_type):
        HTTPException.__init__(self, 'got more than %d bytes when reading %s' % (_MAXLINE, line_type))