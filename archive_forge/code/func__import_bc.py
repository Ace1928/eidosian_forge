import importlib
import re
from ray.rllib.utils.deprecation import Deprecated
def _import_bc():
    import ray.rllib.algorithms.bc as bc
    return (bc.BC, bc.BC.get_default_config())