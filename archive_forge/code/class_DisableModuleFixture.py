import logging
import sys
import urllib.parse as urlparse
import uuid
import fixtures
from oslo_serialization import jsonutils
import requests
import requests_mock
from requests_mock.contrib import fixture
import testscenarios
import testtools
from keystoneclient.tests.unit import client_fixtures
class DisableModuleFixture(fixtures.Fixture):
    """A fixture to provide support for unloading/disabling modules."""

    def __init__(self, module, *args, **kw):
        super(DisableModuleFixture, self).__init__(*args, **kw)
        self.module = module
        self._finders = []
        self._cleared_modules = {}

    def tearDown(self):
        super(DisableModuleFixture, self).tearDown()
        for finder in self._finders:
            sys.meta_path.remove(finder)
        sys.modules.update(self._cleared_modules)

    def clear_module(self):
        cleared_modules = {}
        for fullname in list(sys.modules):
            if fullname == self.module or fullname.startswith(self.module + '.'):
                cleared_modules[fullname] = sys.modules.pop(fullname)
        return cleared_modules

    def setUp(self):
        """Ensure ImportError for the specified module."""
        super(DisableModuleFixture, self).setUp()
        self._cleared_modules.update(self.clear_module())
        finder = NoModuleFinder(self.module)
        self._finders.append(finder)
        sys.meta_path.insert(0, finder)