import operator
import typing
from collections.abc import MappingView, MutableMapping, MutableSet
def _instance_key(self, ref):
    return (id(ref[0]), ref[1])