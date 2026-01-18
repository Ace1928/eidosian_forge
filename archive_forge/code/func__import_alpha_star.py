import importlib
import re
from ray.rllib.utils.deprecation import Deprecated
def _import_alpha_star():
    import ray.rllib.algorithms.alpha_star as alpha_star
    return (alpha_star.AlphaStar, alpha_star.AlphaStar.get_default_config())