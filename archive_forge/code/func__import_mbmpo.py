import importlib
import re
from ray.rllib.utils.deprecation import Deprecated
def _import_mbmpo():
    import ray.rllib.algorithms.mbmpo as mbmpo
    return (mbmpo.MBMPO, mbmpo.MBMPO.get_default_config())