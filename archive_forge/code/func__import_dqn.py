import importlib
import re
from ray.rllib.utils.deprecation import Deprecated
def _import_dqn():
    import ray.rllib.algorithms.dqn as dqn
    return (dqn.DQN, dqn.DQN.get_default_config())