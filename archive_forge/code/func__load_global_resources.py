from stevedore import extension
from heat.common import pluginutils
from heat.engine import clients
from heat.engine import environment
from heat.engine import plugin_manager
def _load_global_resources(env):
    _register_constraints(env, _get_mapping('heat.constraints'))
    _register_stack_lifecycle_plugins(env, _get_mapping('heat.stack_lifecycle_plugins'))
    _register_event_sinks(env, _get_mapping('heat.event_sinks'))
    manager = plugin_manager.PluginManager(__name__)
    resource_mapping = plugin_manager.PluginMapping(['available_resource', 'resource'])
    constraint_mapping = plugin_manager.PluginMapping('constraint')
    _register_resources(env, resource_mapping.load_all(manager))
    _register_constraints(env, constraint_mapping.load_all(manager))