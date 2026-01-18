import importlib
import re
from ray.rllib.utils.deprecation import Deprecated
def _import_maddpg():
    import ray.rllib.algorithms.maddpg as maddpg
    return (maddpg.MADDPG, maddpg.MADDPG.get_default_config())