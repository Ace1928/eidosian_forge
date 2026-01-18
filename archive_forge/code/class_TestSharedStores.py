import os
import sys
import threading
from io import BytesIO
from textwrap import dedent
import configobj
from testtools import matchers
from .. import (bedding, branch, config, controldir, diff, errors, lock,
from .. import registry as _mod_registry
from .. import tests, trace
from .. import transport as _mod_transport
from .. import ui, urlutils
from ..bzr import remote
from ..transport import remote as transport_remote
from . import features, scenarios, test_server
class TestSharedStores(tests.TestCaseInTempDir):

    def test_breezy_conf_shared(self):
        g1 = config.GlobalStack()
        g2 = config.GlobalStack()
        self.assertIs(g1.store, g2.store)