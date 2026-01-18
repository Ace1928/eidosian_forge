import logging
import re
from hashlib import md5
import urllib.parse
import cherrypy
from cherrypy._cpcompat import text_or_bytes
from cherrypy.lib import httputil as _httputil
from cherrypy.lib import is_iterator
Assert username. Raise redirect, or return True if request handled.
        