import time
import functools
from hashlib import md5
from urllib.request import parse_http_list, parse_keqv_list
import cherrypy
from cherrypy._cpcompat import ntob, tonative
def _get_charset_declaration(charset):
    global FALLBACK_CHARSET
    charset = charset.upper()
    return ', charset="%s"' % charset if charset != FALLBACK_CHARSET else ''