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
class TestConfigGetSections(tests.TestCaseWithTransport):

    def setUp(self):
        super().setUp()
        create_configs(self)

    def assertSectionNames(self, expected, conf, name=None):
        """Check which sections are returned for a given config.

        If fallback configurations exist their sections can be included.

        :param expected: A list of section names.

        :param conf: The configuration that will be queried.

        :param name: An optional section name that will be passed to
            get_sections().
        """
        sections = list(conf._get_sections(name))
        self.assertLength(len(expected), sections)
        self.assertEqual(expected, [n for n, _, _ in sections])

    def test_breezy_default_section(self):
        self.assertSectionNames(['DEFAULT'], self.breezy_config)

    def test_locations_default_section(self):
        self.assertSectionNames([], self.locations_config)

    def test_locations_named_section(self):
        self.locations_config.set_user_option('file', 'locations')
        self.assertSectionNames([self.tree.basedir], self.locations_config)

    def test_locations_matching_sections(self):
        loc_config = self.locations_config
        loc_config.set_user_option('file', 'locations')
        parser = loc_config._get_parser()
        location_names = self.tree.basedir.split('/')
        parent = '/'.join(location_names[:-1])
        child = '/'.join(location_names + ['child'])
        parser[parent] = {}
        parser[parent]['file'] = 'parent'
        parser[child] = {}
        parser[child]['file'] = 'child'
        self.assertSectionNames([self.tree.basedir, parent], loc_config)

    def test_branch_data_default_section(self):
        self.assertSectionNames([None], self.branch_config._get_branch_data_config())

    def test_branch_default_sections(self):
        self.assertSectionNames([None, 'DEFAULT'], self.branch_config)
        self.branch_config._get_location_config().set_user_option('file', 'locations')
        self.assertSectionNames([self.tree.basedir, None, 'DEFAULT'], self.branch_config)

    def test_breezy_named_section(self):
        self.breezy_config.set_alias('breezy', 'bzr')
        self.assertSectionNames(['ALIASES'], self.breezy_config, 'ALIASES')