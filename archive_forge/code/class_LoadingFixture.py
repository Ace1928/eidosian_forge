import uuid
import fixtures
from keystoneauth1 import discover
from keystoneauth1 import loading
from keystoneauth1 import plugin
class LoadingFixture(fixtures.Fixture):
    """A fixture that will stub out all plugin loading calls.

    When using keystoneauth plugins loaded from config, CLI or elsewhere it is
    often difficult to handle the plugin parts in tests because we don't have a
    reasonable default.

    This fixture will create a :py:class:`TestPlugin` that will be
    returned for all calls to plugin loading so you can simply bypass the
    authentication steps and return something well known.

    :param str token: The token to include in authenticated requests.
    :param str endpoint: The endpoint to respond to service lookups with.
    :param str user_id: The user_id to report for the authenticated user.
    :param str project_id: The project_id to report for the authenticated user.
    """
    MOCK_POINT = 'keystoneauth1.loading.base.get_plugin_loader'

    def __init__(self, token=None, endpoint=None, user_id=None, project_id=None):
        super(LoadingFixture, self).__init__()
        self.token = token or uuid.uuid4().hex
        self.endpoint = endpoint or DEFAULT_TEST_ENDPOINT
        self.user_id = user_id or uuid.uuid4().hex
        self.project_id = project_id or uuid.uuid4().hex

    def setUp(self):
        super(LoadingFixture, self).setUp()
        self.useFixture(fixtures.MonkeyPatch(self.MOCK_POINT, self.get_plugin_loader))

    def create_plugin(self):
        return TestPlugin(token=self.token, endpoint=self.endpoint, user_id=self.user_id, project_id=self.project_id)

    def get_plugin_loader(self, auth_type):
        plugin = self.create_plugin()
        plugin.auth_type = auth_type
        return _TestPluginLoader(plugin)

    def get_endpoint(self, path=None, **kwargs):
        """Utility function to get the endpoint the plugin would return.

        This function is provided as a convenience so you can do comparisons in
        your tests. Overriding it will not affect the endpoint returned by the
        plugin.

        :param str path: The path to append to the plugin endpoint.
        """
        endpoint = _format_endpoint(self.endpoint, **kwargs)
        if path:
            endpoint = '%s/%s' % (endpoint.rstrip('/'), path.lstrip('/'))
        return endpoint