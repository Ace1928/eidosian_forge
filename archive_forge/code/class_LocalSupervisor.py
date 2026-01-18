import datetime
import io
import logging
import os
import re
import subprocess
import sys
import time
import unittest
import warnings
import contextlib
import portend
import pytest
from cheroot.test import webtest
import cherrypy
from cherrypy._cpcompat import text_or_bytes, HTTPSConnection, ntob
from cherrypy.lib import httputil
from cherrypy.lib import gctools
class LocalSupervisor(Supervisor):
    """Base class for modeling/controlling servers which run in the same
    process.

    When the server side runs in a different process, start/stop can dump all
    state between each test module easily. When the server side runs in the
    same process as the client, however, we have to do a bit more work to
    ensure config and mounted apps are reset between tests.
    """
    using_apache = False
    using_wsgi = False

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        cherrypy.server.httpserver = self.httpserver_class
        cherrypy.config.update({'log.screen': False})
        engine = cherrypy.engine
        if hasattr(engine, 'signal_handler'):
            engine.signal_handler.subscribe()
        if hasattr(engine, 'console_control_handler'):
            engine.console_control_handler.subscribe()

    def start(self, modulename=None):
        """Load and start the HTTP server."""
        if modulename:
            cherrypy.server.httpserver = None
        cherrypy.engine.start()
        self.sync_apps()

    def sync_apps(self):
        """Tell the server about any apps which the setup functions mounted."""
        pass

    def stop(self):
        td = getattr(self, 'teardown', None)
        if td:
            td()
        cherrypy.engine.exit()
        for name, server in getattr(cherrypy, 'servers', {}).copy().items():
            server.unsubscribe()
            del cherrypy.servers[name]