from contextlib import contextmanager
import os
import shutil
import socket
import stat
import tempfile
import unittest
import warnings
from lazr.restfulclient.resource import ServiceRoot
from launchpadlib.credentials import (
from launchpadlib import uris
import launchpadlib.launchpad
from launchpadlib.launchpad import Launchpad
from launchpadlib.credentials import UnencryptedFileCredentialStore
from launchpadlib.testing.helpers import (
from launchpadlib.credentials import (
class TestServiceNameWithEmbeddedVersion(unittest.TestCase):
    """Reject service roots that include the version at the end of the URL.

    If the service root is "http://api.launchpad.net/beta/" and the
    version is "beta", the launchpadlib constructor will raise an
    exception.

    This happens with scripts that were written against old versions
    of launchpadlib. The alternative is to try to silently fix it (the
    fix will eventually break as new versions of the web service are
    released) or to go ahead and make a request to
    http://api.launchpad.net/beta/beta/, and cause an unhelpful 404
    error.
    """

    def test_service_name_with_embedded_version(self):
        version = 'version-foo'
        root = uris.service_roots['staging'] + version
        try:
            Launchpad(None, None, None, service_root=root, version=version)
        except ValueError as e:
            self.assertTrue(str(e).startswith('It looks like you\'re using a service root that incorporates the name of the web service version ("version-foo")'))
        else:
            raise AssertionError('Expected a ValueError that was not thrown!')
        root += '/'
        self.assertRaises(ValueError, Launchpad, None, None, None, service_root=root, version=version)
        default_version = NoNetworkLaunchpad.DEFAULT_VERSION
        root = uris.service_roots['staging'] + default_version + '/'
        self.assertRaises(ValueError, Launchpad, None, None, None, service_root=root)