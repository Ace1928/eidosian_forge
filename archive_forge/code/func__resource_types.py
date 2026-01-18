from . import schema
from .jsonutil import get_column
from .search import Search
def _resource_types(self, name):
    return list(set(self._resource_struct(name).values()))