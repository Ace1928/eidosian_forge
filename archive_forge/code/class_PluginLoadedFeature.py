import importlib
import os
import stat
import subprocess
import sys
import tempfile
import warnings
from .. import osutils, symbol_versioning
class PluginLoadedFeature(Feature):
    """Check whether a plugin with specific name is loaded.

    This is different from ModuleAvailableFeature, because
    plugins can be available but explicitly disabled
    (e.g. through BRZ_DISABLE_PLUGINS=blah).

    :ivar plugin_name: The name of the plugin
    """

    def __init__(self, plugin_name):
        super().__init__()
        self.plugin_name = plugin_name

    def _probe(self):
        from breezy.plugin import get_loaded_plugin
        return get_loaded_plugin(self.plugin_name) is not None

    @property
    def plugin(self):
        from breezy.plugin import get_loaded_plugin
        return get_loaded_plugin(self.plugin_name)

    def feature_name(self):
        return '%s plugin' % self.plugin_name