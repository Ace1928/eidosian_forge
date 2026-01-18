import os
import sys
import time
import warnings
import contextlib
import portend
class FlupSCGIServer(object):
    """Adapter for a flup.server.scgi.WSGIServer."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.ready = False

    def start(self):
        """Start the SCGI server."""
        from flup.server.scgi import WSGIServer
        self.scgiserver = WSGIServer(*self.args, **self.kwargs)
        self.scgiserver._installSignalHandlers = lambda: None
        self.scgiserver._oldSIGs = []
        self.ready = True
        self.scgiserver.run()

    def stop(self):
        """Stop the HTTP server."""
        self.ready = False
        self.scgiserver._keepGoing = False
        self.scgiserver._threadPool.maxSpare = 0