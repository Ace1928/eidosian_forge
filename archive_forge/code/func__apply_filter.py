import collections.abc
import copy
import functools
from dogpile.cache import region as dp_region
from oslo_utils import uuidutils
from oslo_cache.backends import mongo
from oslo_cache import core
from oslo_cache import exception
from oslo_cache.tests import test_cache
def _apply_filter(self, document, query):
    for key, search in query.items():
        doc_val = document.get(key)
        if isinstance(search, dict):
            op_dict = {'$in': lambda dv, sv: dv in sv}
            is_match = all((op_str in op_dict and op_dict[op_str](doc_val, search_val) for op_str, search_val in search.items()))
        else:
            is_match = doc_val == search
    return is_match