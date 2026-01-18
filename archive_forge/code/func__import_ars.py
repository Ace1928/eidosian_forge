import importlib
import re
from ray.rllib.utils.deprecation import Deprecated
def _import_ars():
    import ray.rllib.algorithms.ars as ars
    return (ars.ARS, ars.ARS.get_default_config())