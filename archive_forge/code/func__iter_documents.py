import collections.abc
import copy
import functools
from dogpile.cache import region as dp_region
from oslo_utils import uuidutils
from oslo_cache.backends import mongo
from oslo_cache import core
from oslo_cache import exception
from oslo_cache.tests import test_cache
def _iter_documents(self, spec=None):
    return (SON_MANIPULATOR.transform_outgoing(document, self) for document in self._documents.values() if self._apply_filter(document, spec))