from threading import local as _local
from ._cperror import (
from . import _cpdispatch as dispatch
from ._cptools import default_toolbox as tools, Tool
from ._helper import expose, popargs, url
from . import _cprequest, _cpserver, _cptree, _cplogging, _cpconfig
import cherrypy.lib.httputil as _httputil
from ._cptree import Application
from . import _cpwsgi as wsgi
from . import process
from . import _cpchecker
def _cherrypy_pydoc_resolve(thing, forceload=0):
    """Given an object or a path to an object, get the object and its name."""
    if isinstance(thing, _ThreadLocalProxy):
        thing = getattr(serving, thing.__attrname__)
    return _pydoc._builtin_resolve(thing, forceload)