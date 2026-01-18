import importlib
import re
from ray.rllib.utils.deprecation import Deprecated
def _import_rnnsac():
    from ray.rllib.algorithms import sac
    return (sac.RNNSAC, sac.RNNSAC.get_default_config())