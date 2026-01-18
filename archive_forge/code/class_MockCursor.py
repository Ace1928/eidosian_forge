import collections.abc
import copy
import functools
from dogpile.cache import region as dp_region
from oslo_utils import uuidutils
from oslo_cache.backends import mongo
from oslo_cache import core
from oslo_cache import exception
from oslo_cache.tests import test_cache
class MockCursor(object):

    def __init__(self, collection, dataset_factory):
        super(MockCursor, self).__init__()
        self.collection = collection
        self._factory = dataset_factory
        self._dataset = self._factory()
        self._limit = None
        self._skip = None

    def __iter__(self):
        return self

    def __next__(self):
        if self._skip:
            for _ in range(self._skip):
                next(self._dataset)
            self._skip = None
        if self._limit is not None and self._limit <= 0:
            raise StopIteration()
        if self._limit is not None:
            self._limit -= 1
        return next(self._dataset)
    next = __next__

    def __getitem__(self, index):
        arr = [x for x in self._dataset]
        self._dataset = iter(arr)
        return arr[index]