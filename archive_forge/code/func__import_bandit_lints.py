import importlib
import re
from ray.rllib.utils.deprecation import Deprecated
def _import_bandit_lints():
    from ray.rllib.algorithms.bandit.bandit import BanditLinTS
    return (BanditLinTS, BanditLinTS.get_default_config())