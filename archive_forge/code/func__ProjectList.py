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
def _ProjectList(self, obj, projection, flag):
    """Projects a list, tuple or set object.

    Args:
      obj: A list, tuple or set.
      projection: Projection _Tree node.
      flag: A bitmask of DEFAULT, INNER, PROJECT.

    Returns:
      The projected obj.
    """
    if obj is None:
        return None
    if not obj:
        return []
    try:
        _ = len(obj)
        try:
            _ = obj[0]
        except TypeError:
            obj = sorted(obj)
    except TypeError:
        try:
            obj = list(obj)
        except TypeError:
            return None
    indices = set([])
    sliced = None
    if not projection.tree:
        if flag < self._projection.PROJECT:
            return None
    else:
        for index in projection.tree:
            if index is None:
                if flag >= self._projection.PROJECT or projection.tree[index].attribute.flag:
                    sliced = projection.tree[index]
            elif isinstance(index, six.integer_types) and index in range(-len(obj), len(obj)):
                indices.add(index)
    if flag >= self._projection.PROJECT and (not sliced):
        sliced = self._projection.GetEmpty()
    if not indices and (not sliced):
        return None
    maxindex = -1
    if sliced:
        res = [None] * len(obj)
    else:
        res = [None] * (max((x + len(obj) if x < 0 else x for x in indices)) + 1)
    for index in range(len(obj)) if sliced else indices:
        val = obj[index]
        if val is None:
            continue
        f = flag
        if index in projection.tree:
            child_projection = projection.tree[index]
            if sliced:
                f |= sliced.attribute.flag
        else:
            child_projection = sliced
        if child_projection:
            f |= child_projection.attribute.flag
            if f >= self._projection.INNER:
                val = self._Project(val, child_projection, f)
            else:
                val = None
        if val is None:
            continue
        if index < 0:
            index += len(obj)
        if maxindex < index:
            maxindex = index
        res[index] = val
    if maxindex < 0:
        return None
    return res[0:maxindex + 1] if sliced else res