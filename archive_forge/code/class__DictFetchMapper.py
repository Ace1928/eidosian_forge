import collections
import functools
import re
import threading
import warnings
import numpy as np
import wrapt
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import pywrap_tf_session as tf_session
from tensorflow.python.eager import context
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import device
from tensorflow.python.framework import error_interpolation
from tensorflow.python.framework import errors
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import stack
from tensorflow.python.framework import tensor
from tensorflow.python.ops import session_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.experimental import mixed_precision_global_state
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
class _DictFetchMapper(_FetchMapper):
    """Fetch mapper for dicts."""

    def __init__(self, fetches):
        """Creates a _DictFetchMapper.

    Args:
      fetches: Dict of fetches.
    """
        self._fetch_type = type(fetches)
        if isinstance(fetches, collections.defaultdict):
            self._type_ctor = functools.partial(collections.defaultdict, fetches.default_factory)
        else:
            self._type_ctor = self._fetch_type
        self._keys = fetches.keys()
        self._mappers = [_FetchMapper.for_fetch(fetch) for fetch in fetches.values()]
        self._unique_fetches, self._value_indices = _uniquify_fetches(self._mappers)

    def unique_fetches(self):
        return self._unique_fetches

    def build_results(self, values):

        def _generator():
            for k, m, vi in zip(self._keys, self._mappers, self._value_indices):
                yield (k, m.build_results([values[j] for j in vi]))
        return self._type_ctor(_generator())