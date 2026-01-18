from typing import Callable, Dict, Type
import importlib
from collections import namedtuple
def _name_to_agent_class(name: str):
    """
    Convert agent name to class.

    This adds "Agent" to the end of the name and uppercases the first letter
    and the first letter appearing after each underscore (underscores are
    removed).

    :param name:
        name of agent, e.g. local_human

    :return:
        class of agent, e.g. LocalHumanAgent.
    """
    words = name.split('_')
    class_name = ''
    for w in words:
        class_name += w[0].upper() + w[1:]
    class_name += 'Agent'
    return class_name