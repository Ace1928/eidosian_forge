import os
import random
import sys
import time
import xml.sax
import xml.sax.handler
from io import StringIO
from breezy import errors, osutils, trace, transport
from breezy.transport.http import urllib
class DavConnectionHandler(urllib.ConnectionHandler):
    """Custom connection handler.

    We need to use the DavConnectionHTTPxConnection class to take
    into account our own DavResponse objects, to be able to
    declare our own body ignored responses, sigh.
    """

    def http_request(self, request):
        return self.capture_connection(request, DavHTTPConnection)

    def https_request(self, request):
        return self.capture_connection(request, DavHTTPSConnection)