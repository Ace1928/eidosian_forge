from stevedore import extension
from heat.common import pluginutils
from heat.engine import clients
from heat.engine import environment
from heat.engine import plugin_manager
def _load_global_environment(env):
    _load_global_resources(env)
    environment.read_global_environment(env)