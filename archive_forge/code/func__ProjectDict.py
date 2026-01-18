from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import json
from apitools.base.protorpclite import messages as protorpc_message
from apitools.base.py import encoding as protorpc_encoding
from googlecloudsdk.core.resource import resource_projection_parser
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import encoding
import six
from six.moves import range  # pylint: disable=redefined-builtin
def _ProjectDict(self, obj, projection, flag):
    """Projects a dictionary object.

    Args:
      obj: A dict.
      projection: Projection _Tree node.
      flag: A bitmask of DEFAULT, INNER, PROJECT.

    Returns:
      The projected obj.
    """
    if not obj:
        return obj
    res = {}
    try:
        six.iteritems(obj)
    except ValueError:
        return None
    for key, val in six.iteritems(obj):
        f = flag
        if key in projection.tree:
            child_projection = projection.tree[key]
            f |= child_projection.attribute.flag
            if f < self._projection.INNER:
                continue
            val = self._Project(val, child_projection, f)
        else:
            val = self._ProjectAttribute(val, self._projection.GetEmpty(), f)
        if val is not None or self._retain_none_values or (f >= self._projection.PROJECT and self._columns):
            try:
                res[encoding.Decode(key)] = val
            except UnicodeError:
                res[key] = val
    return res or None