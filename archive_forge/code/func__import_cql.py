import importlib
import re
from ray.rllib.utils.deprecation import Deprecated
def _import_cql():
    import ray.rllib.algorithms.cql as cql
    return (cql.CQL, cql.CQL.get_default_config())