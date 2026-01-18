import fixtures
from keystone import auth
import keystone.server
class BackendLoader(fixtures.Fixture):
    """Initialize each manager and assigns them to an attribute."""

    def __init__(self, testcase):
        super(BackendLoader, self).__init__()
        self._testcase = testcase

    def setUp(self):
        super(BackendLoader, self).setUp()
        self.clear_auth_plugin_registry()
        drivers, _unused = keystone.server.setup_backends()
        for manager_name, manager in drivers.items():
            setattr(self._testcase, manager_name, manager)
        self.addCleanup(self._testcase.cleanup_instance(*list(drivers.keys())))
        del self._testcase

    def clear_auth_plugin_registry(self):
        auth.core.AUTH_METHODS.clear()
        auth.core.AUTH_PLUGINS_LOADED = False