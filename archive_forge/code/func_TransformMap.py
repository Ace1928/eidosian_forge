from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import datetime
import io
import re
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.resource import resource_exceptions
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import times
import six
from six.moves import map  # pylint: disable=redefined-builtin
from six.moves import urllib
def TransformMap(r, depth=1):
    """Applies the next transform in the sequence to each resource list item.

  Example:
    ```list_field.map().foo().list()```:::
    Applies foo() to each item in list_field and then list() to the resulting
    value to return a compact comma-separated list.
    ```list_field.*foo().list()```:::
    ```*``` is shorthand for map().
    ```list_field.map().foo().map().bar()```:::
    Applies foo() to each item in list_field and then bar() to each item in the
    resulting list.
    ```abc.xyz.map(2).foo()```:::
    Applies foo() to each item in xyz[] for all items in abc[].
    ```abc.xyz.**foo()```:::
    ```**``` is shorthand for map(2).

  Args:
    r: A resource.
    depth: The list nesting depth.

  Returns:
    r.
  """
    _ = depth
    return r