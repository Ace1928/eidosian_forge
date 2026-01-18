import re
import sys
import tempfile
from urllib.parse import unquote
import cheroot.server
import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy.lib import httputil
def _old_process_multipart(entity):
    """The behavior of 3.2 and lower. Deprecated and will be changed in 3.3."""
    process_multipart(entity)
    params = entity.params
    for part in entity.parts:
        if part.name is None:
            key = ntou('parts')
        else:
            key = part.name
        if part.filename is None:
            value = part.fullvalue()
        else:
            value = part
        if key in params:
            if not isinstance(params[key], list):
                params[key] = [params[key]]
            params[key].append(value)
        else:
            params[key] = value