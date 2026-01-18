import json
from string import Template
import re
import sys
from webob.acceptparse import create_accept_header
from webob.compat import (
from webob.request import Request
from webob.response import Response
from webob.util import html_escape
class HTTPLocked(HTTPClientError):
    """
    subclass of :class:`~HTTPClientError`

    This indicates that the resource is locked.

    code: 423, title: Locked
    """
    code = 423
    title = 'Locked'
    explanation = 'The resource is locked'