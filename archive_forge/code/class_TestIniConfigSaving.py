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
class TestIniConfigSaving(tests.TestCaseInTempDir):

    def test_cant_save_without_a_file_name(self):
        conf = config.IniBasedConfig()
        self.assertRaises(AssertionError, conf._write_config_file)

    def test_saved_with_content(self):
        content = b'foo = bar\n'
        config.IniBasedConfig.from_string(content, file_name='./test.conf', save=True)
        self.assertFileEqual(content, 'test.conf')