import abc
import collections
import inspect
import sys
import uuid
import random
from .._utils import patch_collections_abc, stringify_id, OrderedSet
@staticmethod
def _id_str(component):
    id_ = stringify_id(getattr(component, 'id', ''))
    return id_ and f' (id={id_:s})'