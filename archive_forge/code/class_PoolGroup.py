from __future__ import annotations
import os
from itertools import chain
from .connection import Resource
from .messaging import Producer
from .utils.collections import EqualityDict
from .utils.compat import register_after_fork
from .utils.functional import lazy
class PoolGroup(EqualityDict):
    """Collection of resource pools."""

    def __init__(self, limit=None, close_after_fork=True):
        self.limit = limit
        self.close_after_fork = close_after_fork
        if self.close_after_fork and register_after_fork is not None:
            register_after_fork(self, _after_fork_cleanup_group)

    def create(self, resource, limit):
        raise NotImplementedError('PoolGroups must define ``create``')

    def __missing__(self, resource):
        limit = self.limit
        if limit is use_global_limit:
            limit = get_limit()
        k = self[resource] = self.create(resource, limit)
        return k