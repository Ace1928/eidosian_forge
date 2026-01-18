import socketserver
from .. import errors, tests
from ..bzr.tests import test_read_bundle
from ..directory_service import directories
from ..mergeable import read_mergeable_from_url
from . import test_server
class DisconnectingHandler(socketserver.BaseRequestHandler):
    """A request handler that immediately closes any connection made to it."""

    def handle(self):
        self.request.close()