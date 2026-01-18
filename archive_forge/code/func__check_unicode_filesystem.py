import io
import os
import sys
import re
import platform
import tempfile
import urllib.parse
import unittest.mock
from http.client import HTTPConnection
import pytest
import py.path
import path
import cherrypy
from cherrypy.lib import static
from cherrypy._cpcompat import HTTPSConnection, ntou, tonative
from cherrypy.test import helper
def _check_unicode_filesystem(tmpdir):
    filename = tmpdir / ntou('☃', 'utf-8')
    tmpl = 'File system encoding ({encoding}) cannot support unicode filenames'
    msg = tmpl.format(encoding=sys.getfilesystemencoding())
    try:
        io.open(str(filename), 'w').close()
    except UnicodeEncodeError:
        pytest.skip(msg)