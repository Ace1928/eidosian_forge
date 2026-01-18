import os
import sys
from breezy import branch, osutils, registry, tests
class InvasiveGetter(registry._ObjectGetter):

    def get_obj(inner_self):
        _registry.register('more hidden', None)
        return inner_self._obj