from collections import defaultdict
import hashlib
from typing import Any, Dict, Tuple
from ray.tune.search.sample import Categorical, Domain, Function
from ray.tune.search.variant_generator import assign_value
from ray.util.annotations import DeveloperAPI
class _FunctionResolver:
    """Replaced value for function typed objects."""
    TOKEN = '__fn_ph'

    def __init__(self, hash, fn):
        self.hash = hash
        self._fn = fn

    def resolve(self, config: Dict):
        """Some functions take a resolved spec dict as input.

        Note: Function placeholders are independently sampled during
        resolution. Therefore their random states are not restored.
        """
        return self._fn.sample(config=config)

    def get_placeholder(self) -> str:
        return (self.TOKEN, self.hash)