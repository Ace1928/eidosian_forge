from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import re
from googlecloudsdk.core.resource import resource_exceptions
from googlecloudsdk.core.resource import resource_projection_spec
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.resource import resource_transform
import six
from six.moves import map  # pylint: disable=redefined-builtin
from six.moves import range  # pylint: disable=redefined-builtin
def MakeTransform(func_name, func, args=None, kwargs=None):
    """Returns a transform call object for func(*args, **kwargs).

  Args:
    func_name: The function name.
    func: The function object.
    args: The actual call args.
    kwargs: The actual call kwargs.

  Returns:
    A transform call object for func(obj, *args, **kwargs).
  """
    calls = _Transform()
    calls.Add(_TransformCall(func_name, func, args=args, kwargs=kwargs))
    return calls