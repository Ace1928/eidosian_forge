import re
import sys
import cgi
import os
import os.path
import urllib.parse
import cherrypy
def _percent(statements, missing):
    s = len(statements)
    e = s - len(missing)
    if s > 0:
        return int(round(100.0 * e / s))
    return 0