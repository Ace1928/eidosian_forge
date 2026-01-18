import os
import random
import sys
import time
import xml.sax
import xml.sax.handler
from io import StringIO
from breezy import errors, osutils, trace, transport
from breezy.transport.http import urllib
class DavOpener(urllib.Opener):
    """Dav specific needs regarding HTTP(S)"""

    def __init__(self, report_activity=None, ca_certs=None):
        super().__init__(connection=DavConnectionHandler, report_activity=report_activity, ca_certs=ca_certs)