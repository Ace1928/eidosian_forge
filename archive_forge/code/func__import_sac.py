import importlib
import re
from ray.rllib.utils.deprecation import Deprecated
def _import_sac():
    import ray.rllib.algorithms.sac as sac
    return (sac.SAC, sac.SAC.get_default_config())