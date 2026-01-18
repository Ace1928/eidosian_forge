import pkgutil
import sys
from oslo_concurrency import lockutils
from oslo_log import log as logging
from oslo_utils import importutils
from stevedore import driver
from stevedore import enabled
from neutron_lib._i18n import _
class NamespacedPlugins(object):
    """Wraps a stevedore plugin namespace to load/access its plugins."""

    def __init__(self, namespace):
        self.namespace = namespace
        self._extension_manager = None
        self._extensions = {}
        self.reload()

    def reload(self):
        """Force a reload of the plugins for this instances namespace.

        :returns: None.
        """
        self._extensions = {}
        self._extension_manager = enabled.EnabledExtensionManager(self.namespace, lambda x: True, invoke_on_load=False)
        if not self._extension_manager.names():
            LOG.debug('No plugins found in namespace: ', self.namespace)
            return
        self._extension_manager.map(self._add_extension)

    def _add_extension(self, ext):
        if ext.name in self._extensions:
            msg = _("Plugin '%(p)s' already in namespace: %(ns)s") % {'p': ext.name, 'ns': self.namespace}
            raise KeyError(msg)
        LOG.debug("Loaded plugin '%s' from namespace: %s", ext.name, self.namespace)
        self._extensions[ext.name] = ext

    def _assert_plugin_loaded(self, plugin_name):
        if plugin_name not in self._extensions:
            msg = _("Plugin '%(p)s' not in namespace: %(ns)s") % {'p': plugin_name, 'ns': self.namespace}
            raise KeyError(msg)

    def get_plugin_class(self, plugin_name):
        """Gets a reference to a loaded plugin's class.

        :param plugin_name: The name of the plugin to get the class for.
        :returns: A reference to the loaded plugin's class.
        :raises: KeyError if plugin_name is not loaded.
        """
        self._assert_plugin_loaded(plugin_name)
        return self._extensions[plugin_name].plugin

    def new_plugin_instance(self, plugin_name, *args, **kwargs):
        """Create a new instance of a plugin.

        :param plugin_name: The name of the plugin to instantiate.
        :param args: Any args to pass onto the constructor.
        :param kwargs: Any kwargs to pass onto the constructor.
        :returns: A new instance of plugin_name.
        :raises: KeyError if plugin_name is not loaded.
        """
        self._assert_plugin_loaded(plugin_name)
        return self.get_plugin_class(plugin_name)(*args, **kwargs)

    @property
    def loaded_plugin_names(self):
        return self._extensions.keys()