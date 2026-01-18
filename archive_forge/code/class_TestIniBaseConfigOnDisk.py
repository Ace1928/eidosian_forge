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
class TestIniBaseConfigOnDisk(tests.TestCaseInTempDir):

    def test_cannot_reload_without_name(self):
        conf = config.IniBasedConfig.from_string(sample_config_text)
        self.assertRaises(AssertionError, conf.reload)

    def test_reload_see_new_value(self):
        c1 = config.IniBasedConfig.from_string('editor=vim\n', file_name='./test/conf')
        c1._write_config_file()
        c2 = config.IniBasedConfig.from_string('editor=emacs\n', file_name='./test/conf')
        c2._write_config_file()
        self.assertEqual('vim', c1.get_user_option('editor'))
        self.assertEqual('emacs', c2.get_user_option('editor'))
        c1.reload()
        self.assertEqual('emacs', c1.get_user_option('editor'))