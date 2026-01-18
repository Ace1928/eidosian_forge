import importlib
import re
from ray.rllib.utils.deprecation import Deprecated
def _import_es():
    import ray.rllib.algorithms.es as es
    return (es.ES, es.ES.get_default_config())