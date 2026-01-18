from typing import Callable, Dict, Type
import importlib
from collections import namedtuple
def _get_default_world(default_world=None, num_agents=None):
    """
    Get default world if a world is not already specified by the task.

    If a default world is provided, return this. Otherwise, return
    DialogPartnerWorld if there are 2 agents and MultiAgentDialogWorld if
    there are more.

    :param default_world:
        default world to return
    :param num_agents:
        number of agents in the environment
    """
    if default_world is not None:
        world_class = default_world
    elif num_agents is not None:
        import parlai.core.worlds as core_worlds
        world_name = 'DialogPartnerWorld' if num_agents == 2 else 'MultiAgentDialogWorld'
        world_class = getattr(core_worlds, world_name)
    else:
        return None
    return world_class