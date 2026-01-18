import importlib
import re
from ray.rllib.utils.deprecation import Deprecated
def _import_marwil():
    import ray.rllib.algorithms.marwil as marwil
    return (marwil.MARWIL, marwil.MARWIL.get_default_config())