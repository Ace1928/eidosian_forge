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
def _get_attrs_items(obj):
    """Returns a list of (name, value) pairs from an attrs instance.

  TODO(b/268078256): check if this comment is valid, and if so, ensure it's
  handled in the function below.
  The list will be sorted by name.

  Args:
    obj: an object.

  Returns:
    A list of (attr_name, attr_value) pairs, sorted by attr_name.
  """
    attrs = getattr(obj.__class__, '__attrs_attrs__')
    attr_names = (a.name for a in attrs)
    return [(attr_name, getattr(obj, attr_name)) for attr_name in attr_names]