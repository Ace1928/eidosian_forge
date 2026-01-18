import abc
import collections
import inspect
import sys
import uuid
import random
from .._utils import patch_collections_abc, stringify_id, OrderedSet
class _UNDEFINED:

    def __repr__(self):
        return 'undefined'

    def __str__(self):
        return 'undefined'