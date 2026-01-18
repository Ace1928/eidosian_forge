import time
from . import debug, errors, osutils, revision, trace
class CallableToParentsProviderAdapter:
    """A parents provider that adapts any callable to the parents provider API.

    i.e. it accepts calls to self.get_parent_map and relays them to the
    callable it was constructed with.
    """

    def __init__(self, a_callable):
        self.callable = a_callable

    def __repr__(self):
        return '{}({!r})'.format(self.__class__.__name__, self.callable)

    def get_parent_map(self, keys):
        return self.callable(keys)