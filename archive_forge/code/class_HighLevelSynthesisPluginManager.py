import abc
from typing import List
import stevedore
class HighLevelSynthesisPluginManager:
    """Class tracking the installed high-level-synthesis plugins."""

    def __init__(self):
        self.plugins = stevedore.ExtensionManager('qiskit.synthesis', invoke_on_load=True, propagate_map_exceptions=True)
        self.plugins_by_op = {}
        for plugin_name in self.plugins.names():
            op_name, method_name = plugin_name.split('.')
            if op_name not in self.plugins_by_op.keys():
                self.plugins_by_op[op_name] = []
            self.plugins_by_op[op_name].append(method_name)

    def method_names(self, op_name):
        """Returns plugin methods for op_name."""
        if op_name in self.plugins_by_op.keys():
            return self.plugins_by_op[op_name]
        else:
            return []

    def method(self, op_name, method_name):
        """Returns the plugin for ``op_name`` and ``method_name``."""
        plugin_name = op_name + '.' + method_name
        return self.plugins[plugin_name].obj