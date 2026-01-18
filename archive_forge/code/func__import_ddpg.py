import importlib
import re
from ray.rllib.utils.deprecation import Deprecated
def _import_ddpg():
    import ray.rllib.algorithms.ddpg as ddpg
    return (ddpg.DDPG, ddpg.DDPG.get_default_config())