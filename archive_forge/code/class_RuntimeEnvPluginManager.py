import logging
import os
import json
from abc import ABC
from typing import List, Dict, Optional, Any, Type
from ray._private.runtime_env.context import RuntimeEnvContext
from ray._private.runtime_env.uri_cache import URICache
from ray._private.runtime_env.constants import (
from ray.util.annotations import DeveloperAPI
from ray._private.utils import import_attr
class RuntimeEnvPluginManager:
    """This manager is used to load plugins in runtime env agent."""

    def __init__(self):
        self.plugins: Dict[str, PluginSetupContext] = {}
        plugin_config_str = os.environ.get(RAY_RUNTIME_ENV_PLUGINS_ENV_VAR)
        if plugin_config_str:
            plugin_configs = json.loads(plugin_config_str)
            self.load_plugins(plugin_configs)

    def validate_plugin_class(self, plugin_class: Type[RuntimeEnvPlugin]) -> None:
        if not issubclass(plugin_class, RuntimeEnvPlugin):
            raise RuntimeError(f'Invalid runtime env plugin class {plugin_class}. The plugin class must inherit ray._private.runtime_env.plugin.RuntimeEnvPlugin.')
        if not plugin_class.name:
            raise RuntimeError(f'No valid name in runtime env plugin {plugin_class}.')
        if plugin_class.name in self.plugins:
            raise RuntimeError(f'The name of runtime env plugin {plugin_class} conflicts with {self.plugins[plugin_class.name]}.')

    def validate_priority(self, priority: Any) -> None:
        if not isinstance(priority, int) or priority < RAY_RUNTIME_ENV_PLUGIN_MIN_PRIORITY or priority > RAY_RUNTIME_ENV_PLUGIN_MAX_PRIORITY:
            raise RuntimeError(f'Invalid runtime env priority {priority}, it should be an integer between {RAY_RUNTIME_ENV_PLUGIN_MIN_PRIORITY} and {RAY_RUNTIME_ENV_PLUGIN_MAX_PRIORITY}.')

    def load_plugins(self, plugin_configs: List[Dict]) -> None:
        """Load runtime env plugins and create URI caches for them."""
        for plugin_config in plugin_configs:
            if not isinstance(plugin_config, dict) or RAY_RUNTIME_ENV_CLASS_FIELD_NAME not in plugin_config:
                raise RuntimeError(f'Invalid runtime env plugin config {plugin_config}, it should be a object which contains the {RAY_RUNTIME_ENV_CLASS_FIELD_NAME} field.')
            plugin_class = import_attr(plugin_config[RAY_RUNTIME_ENV_CLASS_FIELD_NAME])
            self.validate_plugin_class(plugin_class)
            if RAY_RUNTIME_ENV_PRIORITY_FIELD_NAME in plugin_config:
                priority = plugin_config[RAY_RUNTIME_ENV_PRIORITY_FIELD_NAME]
            else:
                priority = plugin_class.priority
            self.validate_priority(priority)
            class_instance = plugin_class()
            self.plugins[plugin_class.name] = PluginSetupContext(plugin_class.name, class_instance, priority, self.create_uri_cache_for_plugin(class_instance))

    def add_plugin(self, plugin: RuntimeEnvPlugin) -> None:
        """Add a plugin to the manager and create a URI cache for it.

        Args:
            plugin: The class instance of the plugin.
        """
        plugin_class = type(plugin)
        self.validate_plugin_class(plugin_class)
        self.validate_priority(plugin_class.priority)
        self.plugins[plugin_class.name] = PluginSetupContext(plugin_class.name, plugin, plugin_class.priority, self.create_uri_cache_for_plugin(plugin))

    def create_uri_cache_for_plugin(self, plugin: RuntimeEnvPlugin) -> URICache:
        """Create a URI cache for a plugin.

        Args:
            plugin_name: The name of the plugin.

        Returns:
            The created URI cache for the plugin.
        """
        cache_size_env_var = f'RAY_RUNTIME_ENV_{plugin.name}_CACHE_SIZE_GB'.upper()
        cache_size_bytes = int(1024 ** 3 * float(os.environ.get(cache_size_env_var, 10)))
        return URICache(plugin.delete_uri, cache_size_bytes)

    def sorted_plugin_setup_contexts(self) -> List[PluginSetupContext]:
        """Get the sorted plugin setup contexts, sorted by increasing priority.

        Returns:
            The sorted plugin setup contexts.
        """
        return sorted(self.plugins.values(), key=lambda x: x.priority)