import importlib
import re
from ray.rllib.utils.deprecation import Deprecated
def _import_simple_q():
    import ray.rllib.algorithms.simple_q as simple_q
    return (simple_q.SimpleQ, simple_q.SimpleQ.get_default_config())