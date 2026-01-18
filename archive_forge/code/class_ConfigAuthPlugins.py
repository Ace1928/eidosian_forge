import fixtures
from keystone import auth
import keystone.conf
class ConfigAuthPlugins(fixtures.Fixture):
    """A fixture for setting up and tearing down a auth plugins."""

    def __init__(self, config_fixture, methods, **method_classes):
        super(ConfigAuthPlugins, self).__init__()
        self.methods = methods
        self.config_fixture = config_fixture
        self.method_classes = method_classes

    def setUp(self):
        super(ConfigAuthPlugins, self).setUp()
        if self.methods:
            self.config_fixture.config(group='auth', methods=self.methods)
            keystone.conf.auth.setup_authentication()
        if self.method_classes:
            self.config_fixture.config(group='auth', **self.method_classes)