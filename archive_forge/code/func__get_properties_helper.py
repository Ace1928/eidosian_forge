import pprint
from abc import abstractmethod
@classmethod
def _get_properties_helper(cls):
    return sorted([p for p in cls.__dict__ if isinstance(getattr(cls, p), property)])