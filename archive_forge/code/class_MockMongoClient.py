import collections.abc
import copy
import functools
from dogpile.cache import region as dp_region
from oslo_utils import uuidutils
from oslo_cache.backends import mongo
from oslo_cache import core
from oslo_cache import exception
from oslo_cache.tests import test_cache
class MockMongoClient(object):

    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, dbname):
        return MockMongoDB(dbname)