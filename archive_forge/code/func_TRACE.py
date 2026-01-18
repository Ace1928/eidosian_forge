import time
import functools
from hashlib import md5
from urllib.request import parse_http_list, parse_keqv_list
import cherrypy
from cherrypy._cpcompat import ntob, tonative
def TRACE(msg):
    cherrypy.log(msg, context='TOOLS.AUTH_DIGEST')