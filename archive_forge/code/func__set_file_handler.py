import datetime
import logging
import os
import sys
import cherrypy
from cherrypy import _cperror
def _set_file_handler(self, log, filename):
    h = self._get_builtin_handler(log, 'file')
    if filename:
        if h:
            if h.baseFilename != os.path.abspath(filename):
                h.close()
                log.handlers.remove(h)
                self._add_builtin_file_handler(log, filename)
        else:
            self._add_builtin_file_handler(log, filename)
    elif h:
        h.close()
        log.handlers.remove(h)