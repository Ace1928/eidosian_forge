import importlib
import re
from ray.rllib.utils.deprecation import Deprecated
def _import_pg():
    import ray.rllib.algorithms.pg as pg
    return (pg.PG, pg.PG.get_default_config())