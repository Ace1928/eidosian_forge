import mimetypes
import os
import platform
import re
import stat
import unicodedata
import urllib.parse
from email.generator import _make_boundary as make_boundary
from io import UnsupportedOperation
import cherrypy
from cherrypy._cpcompat import ntob
from cherrypy.lib import cptools, file_generator_limited, httputil
def _make_content_disposition(disposition, file_name):
    """Create HTTP header for downloading a file with a UTF-8 filename.

    This function implements the recommendations of :rfc:`6266#appendix-D`.
    See this and related answers: https://stackoverflow.com/a/8996249/2173868.
    """
    ascii_name = unicodedata.normalize('NFKC', file_name).encode('ascii', errors='ignore').decode()
    header = '{}; filename="{}"'.format(disposition, ascii_name)
    if ascii_name != file_name:
        quoted_name = urllib.parse.quote(file_name)
        header += "; filename*=UTF-8''{}".format(quoted_name)
    return header