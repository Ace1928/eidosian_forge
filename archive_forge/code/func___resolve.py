from collections import defaultdict
import hashlib
from typing import Any, Dict, Tuple
from ray.tune.search.sample import Categorical, Domain, Function
from ray.tune.search.variant_generator import assign_value
from ray.util.annotations import DeveloperAPI
def __resolve(resolver_type, args):
    for path, resolvers in replaced.items():
        assert resolvers
        if not isinstance(resolvers[0], resolver_type):
            continue
        prefix, ph = _get_placeholder(config, (), path)
        if not ph:
            continue
        for resolver in resolvers:
            if resolver.hash != ph[1]:
                continue
            assign_value(config, prefix, resolver.resolve(*args))