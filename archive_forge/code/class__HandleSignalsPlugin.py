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
class _HandleSignalsPlugin(object):
    """Handle signals from other processes.

    Based on the configured platform handlers above.
    """

    def __init__(self, bus):
        self.bus = bus

    def subscribe(self):
        """Add the handlers based on the platform."""
        if hasattr(self.bus, 'signal_handler'):
            self.bus.signal_handler.subscribe()
        if hasattr(self.bus, 'console_control_handler'):
            self.bus.console_control_handler.subscribe()