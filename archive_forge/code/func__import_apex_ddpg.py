import importlib
import re
from ray.rllib.utils.deprecation import Deprecated
def _import_apex_ddpg():
    import ray.rllib.algorithms.apex_ddpg as apex_ddpg
    return (apex_ddpg.ApexDDPG, apex_ddpg.ApexDDPG.get_default_config())