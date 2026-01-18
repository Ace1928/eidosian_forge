import heapq
import inspect
import unittest
from pbr.version import VersionInfo
def _setResource(self, new_resource):
    """Set the current resource to a new value."""
    self._currentResource = new_resource
    self._dirty = False