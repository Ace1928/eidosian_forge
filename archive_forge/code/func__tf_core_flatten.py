import collections as _collections
import enum
import typing
from typing import Protocol
import six as _six
import wrapt as _wrapt
from tensorflow.python import pywrap_tensorflow  # pylint: disable=unused-import
from tensorflow.python.platform import tf_logging
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util.compat import collections_abc as _collections_abc
def _tf_core_flatten(structure, expand_composites=False):
    """See comments for flatten() in tensorflow/python/util/nest.py."""
    if structure is None:
        return [None]
    expand_composites = bool(expand_composites)
    return _pywrap_utils.Flatten(structure, expand_composites)