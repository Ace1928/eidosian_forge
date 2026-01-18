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
class TestIniConfigBuilding(TestIniConfig):

    def test_contructs(self):
        config.IniBasedConfig()

    def test_from_fp(self):
        my_config = config.IniBasedConfig.from_string(sample_config_text)
        self.assertIsInstance(my_config._get_parser(), configobj.ConfigObj)

    def test_cached(self):
        my_config = config.IniBasedConfig.from_string(sample_config_text)
        parser = my_config._get_parser()
        self.assertTrue(my_config._get_parser() is parser)

    def _dummy_chown(self, path, uid, gid):
        self.path, self.uid, self.gid = (path, uid, gid)

    def test_ini_config_ownership(self):
        """Ensure that chown is happening during _write_config_file"""
        self.requireFeature(features.chown_feature)
        self.overrideAttr(os, 'chown', self._dummy_chown)
        self.path = self.uid = self.gid = None
        conf = config.IniBasedConfig(file_name='./foo.conf')
        conf._write_config_file()
        self.assertEqual(self.path, './foo.conf')
        self.assertTrue(isinstance(self.uid, int))
        self.assertTrue(isinstance(self.gid, int))