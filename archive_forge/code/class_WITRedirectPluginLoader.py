from tensorboard.plugins import base_plugin
class WITRedirectPluginLoader(base_plugin.TBLoader):
    """Load the redirect notice iff the dynamic plugin is unavailable."""

    def load(self, context):
        try:
            import tensorboard_plugin_wit
            return None
        except ImportError:
            return _WITRedirectPlugin(context)