import datetime
import logging
import os
import sys
import cherrypy
from cherrypy import _cperror
def _add_builtin_file_handler(self, log, fname):
    h = logging.FileHandler(fname)
    h.setFormatter(logfmt)
    h._cpbuiltin = 'file'
    log.addHandler(h)