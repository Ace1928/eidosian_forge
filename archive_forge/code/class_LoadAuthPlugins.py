import fixtures
from keystone import auth
import keystone.conf
class LoadAuthPlugins(fixtures.Fixture):

    def __init__(self, *method_names):
        super(LoadAuthPlugins, self).__init__()
        self.method_names = method_names
        self.saved = {}

    def setUp(self):
        super(LoadAuthPlugins, self).setUp()
        AUTH_METHODS = auth.core.AUTH_METHODS
        for method_name in self.method_names:
            if method_name in AUTH_METHODS:
                self.saved[method_name] = AUTH_METHODS[method_name]
            AUTH_METHODS[method_name] = auth.core.load_auth_method(method_name)
        auth.core.AUTH_PLUGINS_LOADED = True

    def cleanUp(self):
        AUTH_METHODS = auth.core.AUTH_METHODS
        for method_name in list(AUTH_METHODS):
            if method_name in self.saved:
                AUTH_METHODS[method_name] = self.saved[method_name]
            else:
                del AUTH_METHODS[method_name]
        auth.core.AUTH_PLUGINS_LOADED = False