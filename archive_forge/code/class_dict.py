import itertools
from future.utils import with_metaclass
from .util import subvals
from .extend import (Box, primitive, notrace_primitive, VSpace, vspace,
class dict(with_metaclass(DictMeta, dict_)):

    def __new__(cls, *args, **kwargs):
        result = dict_(*args, **kwargs)
        if result:
            return _make_dict(result.keys(), list(result.values()))
        return result