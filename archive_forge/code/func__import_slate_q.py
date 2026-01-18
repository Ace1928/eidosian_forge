import importlib
import re
from ray.rllib.utils.deprecation import Deprecated
def _import_slate_q():
    import ray.rllib.algorithms.slateq as slateq
    return (slateq.SlateQ, slateq.SlateQ.get_default_config())