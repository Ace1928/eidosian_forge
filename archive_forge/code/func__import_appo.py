import importlib
import re
from ray.rllib.utils.deprecation import Deprecated
def _import_appo():
    import ray.rllib.algorithms.appo as appo
    return (appo.APPO, appo.APPO.get_default_config())