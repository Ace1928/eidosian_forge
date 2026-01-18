from __future__ import annotations
from itertools import count
from .imports import symbol_by_name
class FairCycle:
    """Cycle between resources.

    Consume from a set of resources, where each resource gets
    an equal chance to be consumed from.

    Arguments:
    ---------
        fun (Callable): Callback to call.
        resources (Sequence[Any]): List of resources.
        predicate (type): Exception predicate.
    """

    def __init__(self, fun, resources, predicate=Exception):
        self.fun = fun
        self.resources = resources
        self.predicate = predicate
        self.pos = 0

    def _next(self):
        while 1:
            try:
                resource = self.resources[self.pos]
                self.pos += 1
                return resource
            except IndexError:
                self.pos = 0
                if not self.resources:
                    raise self.predicate()

    def get(self, callback, **kwargs):
        """Get from next resource."""
        for tried in count(0):
            resource = self._next()
            try:
                return self.fun(resource, callback, **kwargs)
            except self.predicate:
                if tried >= len(self.resources) - 1:
                    raise

    def close(self):
        """Close cycle."""

    def __repr__(self):
        """``repr(cycle)``."""
        return '<FairCycle: {self.pos}/{size} {self.resources}>'.format(self=self, size=len(self.resources))