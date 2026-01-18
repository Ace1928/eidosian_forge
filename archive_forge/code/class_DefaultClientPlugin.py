from heat.engine.clients import client_plugin
class DefaultClientPlugin(client_plugin.ClientPlugin):
    """A ClientPlugin that has no client.

    This is provided so that Resource can make use of the is_not_found() and
    is_conflict() methods even if the resource plugin has not specified a
    client plugin.
    """

    def _create(self, version=None):
        return super(DefaultClientPlugin, self)._create(version)