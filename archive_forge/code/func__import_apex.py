import importlib
import re
from ray.rllib.utils.deprecation import Deprecated
def _import_apex():
    import ray.rllib.algorithms.apex_dqn as apex_dqn
    return (apex_dqn.ApexDQN, apex_dqn.ApexDQN.get_default_config())