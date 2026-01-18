from collections import defaultdict
import hashlib
from typing import Any, Dict, Tuple
from ray.tune.search.sample import Categorical, Domain, Function
from ray.tune.search.variant_generator import assign_value
from ray.util.annotations import DeveloperAPI
class _RefResolver:
    """Replaced value for all other non-primitive objects."""
    TOKEN = '__ref_ph'

    def __init__(self, hash, value):
        self.hash = hash
        self._value = value

    def resolve(self):
        return self._value

    def get_placeholder(self) -> str:
        return (self.TOKEN, self.hash)