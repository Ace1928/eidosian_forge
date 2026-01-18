from tensorboard.plugins import base_plugin
class ProfileRedirectPluginLoader(base_plugin.TBLoader):
    """Load the redirect notice iff the dynamic plugin is unavailable."""

    def load(self, context):
        try:
            import tensorboard_plugin_profile
            return None
        except ImportError:
            return _ProfileRedirectPlugin(context)