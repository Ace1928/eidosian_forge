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
class Modality(enum.Enum):
    """Modality/semantic used for treating nested structures.

  - Modality.CORE follows tensorflow_core/tf.nest semantics.

    The following collection types are recognized by `tf.nest` as nested
    structures:

    * `collections.abc.Sequence` (except `string` and `bytes`).
      This includes `list`, `tuple`, and `namedtuple`.
    * `collections.abc.Mapping` (with sortable keys).
      This includes `dict` and `collections.OrderedDict`.
    * `collections.abc.MappingView` (with sortable keys).
    * [`attr.s` classes](https://www.attrs.org/).

    Any other values are considered **atoms**.  Not all collection types are
    considered nested structures.  For example, the following types are
    considered atoms:

    * `set`; `{"a", "b"}` is an atom, while `["a", "b"]` is a nested structure.
    * [`dataclass` classes](https://docs.python.org/library/dataclasses.html)
    * `tf.Tensor`
    * `numpy.array`

  - Modality.DATA follows tf.data's nest semantics.

  This modality makes two changes:
  1. It removes support for lists as a level of nesting in nested structures.
  2. It adds support for `SparseTensorValue` as an atomic element.

  The motivation for this change is twofold:

  1. It seems more natural for lists to be treated (e.g. in Dataset
  constructors)
    as tensors, rather than lists of (lists of...) tensors.
  2. This is needed because `SparseTensorValue` is implemented as a `namedtuple`
    that would normally be flattened and we want to be able to create sparse
    tensor from `SparseTensorValue's similarly to creating tensors from numpy
    arrays.
  """
    CORE = 'CORE'
    DATA = 'DATA'