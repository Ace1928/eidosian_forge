import doctest
import errno
import os
import socket
import subprocess
import sys
import threading
import time
from io import BytesIO
from typing import Optional, Type
from testtools.matchers import DocTestMatches
import breezy
from ... import controldir, debug, errors, osutils, tests
from ... import transport as _mod_transport
from ... import urlutils
from ...tests import features, test_server
from ...transport import local, memory, remote, ssh
from ...transport.http import urllib
from .. import bzrdir
from ..remote import UnknownErrorFromSmartServer
from ..smart import client, medium, message, protocol
from ..smart import request as _mod_request
from ..smart import server as _mod_server
from ..smart import vfs
from . import test_smart
class _InvalidHostnameFeature(features.Feature):
    """Does 'non_existent.invalid' fail to resolve?

    RFC 2606 states that .invalid is reserved for invalid domain names, and
    also underscores are not a valid character in domain names.  Despite this,
    it's possible a badly misconfigured name server might decide to always
    return an address for any name, so this feature allows us to distinguish a
    broken system from a broken test.
    """

    def _probe(self):
        try:
            socket.gethostbyname('non_existent.invalid')
        except socket.gaierror:
            return True
        else:
            return False

    def feature_name(self):
        return 'invalid hostname'