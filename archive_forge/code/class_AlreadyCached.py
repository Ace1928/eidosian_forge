import os
import traceback
from io import StringIO
from twisted import copyright
from twisted.python.compat import execfile, networkString
from twisted.python.filepath import _coerceToFilesystemEncoding
from twisted.web import http, resource, server, static, util
import mygreatresource
class AlreadyCached(Exception):
    """
    This exception is raised when a path has already been cached.
    """