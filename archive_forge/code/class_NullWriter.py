import getopt
import os
import re
import sys
import time
import cherrypy
from cherrypy import _cperror, _cpmodpy
from cherrypy.lib import httputil
class NullWriter(object):
    """Suppresses the printing of socket errors."""

    def write(self, data):
        pass