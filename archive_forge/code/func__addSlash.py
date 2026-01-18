from __future__ import annotations
import errno
import itertools
import mimetypes
import os
import time
import warnings
from html import escape
from typing import Any, Callable, Dict, Sequence
from urllib.parse import quote, unquote
from zope.interface import implementer
from incremental import Version
from typing_extensions import Literal
from twisted.internet import abstract, interfaces
from twisted.python import components, filepath, log
from twisted.python.compat import nativeString, networkString
from twisted.python.deprecate import deprecated
from twisted.python.runtime import platformType
from twisted.python.url import URL
from twisted.python.util import InsensitiveDict
from twisted.web import http, resource, server
from twisted.web.util import redirectTo
def _addSlash(request):
    """
    Add a trailing slash to C{request}'s URI.

    @param request: The incoming request to add the ending slash to.
    @type request: An object conforming to L{twisted.web.iweb.IRequest}

    @return: A URI with a trailing slash, with query and fragment preserved.
    @rtype: L{bytes}
    """
    url = URL.fromText(request.uri.decode('ascii'))
    url = url.replace(path=list(url.path) + [''])
    return url.asText().encode('ascii')