import abc
import collections
import inspect
import sys
import uuid
import random
from .._utils import patch_collections_abc, stringify_id, OrderedSet
class ComponentRegistry:
    """Holds a registry of the namespaces used by components."""
    registry = OrderedSet()
    children_props = collections.defaultdict(dict)
    namespace_to_package = {}

    @classmethod
    def get_resources(cls, resource_name, includes=None):
        resources = []
        for module_name in cls.registry:
            if includes is not None and module_name not in includes:
                continue
            module = sys.modules[module_name]
            resources.extend(getattr(module, resource_name, []))
        return resources