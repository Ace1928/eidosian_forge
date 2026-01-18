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
def _Project(self, obj, projection, flag, leaf=False):
    """Evaluate() helper function.

    This function takes a resource obj and a preprocessed projection. obj
    is a dense subtree of the resource schema (some keys values may be missing)
    and projection is a sparse, possibly improper, subtree of the resource
    schema. Improper in that it may contain paths that do not exist in the
    resource schema or object. _Project() traverses both trees simultaneously,
    guided by the projection tree. When a projection tree path reaches an
    non-existent obj tree path the projection tree traversal is pruned. When a
    projection tree path terminates with an existing obj tree path, that obj
    tree value is projected and the obj tree traversal is pruned.

    Since resources can be sparse a projection can reference values not present
    in a particular resource. Because of this the code is lenient on out of
    bound conditions that would normally be errors.

    Args:
      obj: An object.
      projection: Projection _Tree node.
      flag: A bitmask of DEFAULT, INNER, PROJECT.
      leaf: Do not call _ProjectAttribute() if True.

    Returns:
      An object containing only the key:values selected by projection, or obj if
      the projection is None or empty.
    """
    objid = id(obj)
    if objid in self._been_here_done_that:
        return None
    elif obj is None:
        pass
    elif isinstance(obj, six.text_type) or isinstance(obj, six.binary_type):
        if isinstance(obj, six.binary_type):
            obj = encoding.Decode(obj)
        if self._json_decode and (obj.startswith('{"') and obj.endswith('}') or (obj.startswith('[') and obj.endswith(']'))):
            try:
                return self._Project(json.loads(obj), projection, flag, leaf=leaf)
            except ValueError:
                pass
    elif isinstance(obj, (bool, float, complex)) or isinstance(obj, six.integer_types):
        pass
    elif isinstance(obj, bytearray):
        obj = encoding.Decode(bytes(obj))
    elif isinstance(obj, protorpc_message.Enum):
        obj = obj.name
    else:
        self._been_here_done_that.add(objid)
        from cloudsdk.google.protobuf import message as protobuf_message
        import proto
        if isinstance(obj, protorpc_message.Message):
            obj = protorpc_encoding.MessageToDict(obj)
        elif isinstance(obj, protobuf_message.Message):
            from cloudsdk.google.protobuf import json_format as protobuf_encoding
            obj = protobuf_encoding.MessageToDict(obj)
        elif isinstance(obj, proto.Message):
            obj = obj.__class__.to_dict(obj)
        elif not hasattr(obj, '__iter__') or hasattr(obj, '_fields'):
            obj = self._ProjectClass(obj, projection, flag)
        if projection and projection.attribute and projection.attribute.transform and self._TransformIsEnabled(projection.attribute.transform):
            obj = projection.attribute.transform.Evaluate(obj)
        elif (flag >= self._projection.PROJECT or (projection and projection.tree)) and hasattr(obj, '__iter__'):
            if hasattr(obj, 'items'):
                try:
                    obj = self._ProjectDict(obj, projection, flag)
                except (IOError, TypeError):
                    obj = None
            else:
                try:
                    obj = self._ProjectList(obj, projection, flag)
                except (IOError, TypeError):
                    obj = None
        self._been_here_done_that.discard(objid)
        return obj
    return obj if leaf else self._ProjectAttribute(obj, projection, flag)