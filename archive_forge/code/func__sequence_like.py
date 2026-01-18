import wrapt as _wrapt
from tensorflow.python.util import _pywrap_nest
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import nest_util
from tensorflow.python.util.compat import collections_abc as _collections_abc
from tensorflow.python.util.tf_export import tf_export
@tf_export('__internal__.nest.sequence_like', v1=[])
def _sequence_like(instance, args):
    """Converts the sequence `args` to the same type as `instance`.

  Args:
    instance: an instance of `tuple`, `list`, `namedtuple`, `dict`,
        `collections.OrderedDict`, or `composite_tensor.Composite_Tensor`
        or `type_spec.TypeSpec`.
    args: items to be converted to the `instance` type.

  Returns:
    `args` with the type of `instance`.
  """
    return nest_util.sequence_like(instance, args)