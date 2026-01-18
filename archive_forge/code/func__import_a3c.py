import importlib
import re
from ray.rllib.utils.deprecation import Deprecated
def _import_a3c():
    import ray.rllib.algorithms.a3c as a3c
    return (a3c.A3C, a3c.A3C.get_default_config())