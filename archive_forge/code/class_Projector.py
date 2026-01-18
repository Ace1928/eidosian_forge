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
class Projector(object):
    """Projects a resource using a ProjectionSpec.

  A projector is a method that takes an object and a projection as input and
  produces a new JSON-serializable object containing only the values
  corresponding to the keys in the projection. Optional projection key
  attributes may transform the values in the resulting JSON-serializable object.

  Attributes:
    _projection: The projection object.
    _been_here_done_that: A set of the current object id()'s being projected.
      Used to catch recursive objects like datetime.datetime.max.
    _by_columns: True if Projector projects to a list of columns.
    _columns: self._projection.Columns() column attributes.
    _ignore_default_transforms: Ignore default projection transforms if True.
    _retain_none_values: Retain dict entries with None values.
    _transforms_enabled_attribute: The projection.Attributes()
      transforms_enabled setting.
    _transforms_enabled: Projection attribute transforms enabled if True,
      disabled if False, or set by each Evaluate().
  """

    def __init__(self, projection, by_columns=False, ignore_default_transforms=False, retain_none_values=False):
        """Constructor.

    Args:
      projection: A ProjectionSpec (parsed resource projection expression).
      by_columns: Project to a list of columns if True.
      ignore_default_transforms: Ignore default projection transforms if True.
      retain_none_values: project dict entries with None values.
    """
        self._projection = projection
        self._by_columns = by_columns
        self._columns = self._projection.Columns()
        self._ignore_default_transforms = ignore_default_transforms
        self._retain_none_values = retain_none_values
        self._been_here_done_that = set()
        attributes = projection.Attributes()
        if 'transforms' in attributes:
            self._transforms_enabled_attribute = True
        elif 'no-transforms' in attributes:
            self._transforms_enabled_attribute = False
        else:
            self._transforms_enabled_attribute = None
        self._json_decode = 'json-decode' in attributes

    def _TransformIsEnabled(self, transform):
        """Returns True if transform is enabled.

    Args:
      transform: The resource_projection_parser._Transform object.

    Returns:
      True if the transform is enabled, False if not.
    """
        if self._transforms_enabled is not None:
            return self._transforms_enabled
        return transform.active in (None, self._projection.active)

    def _ProjectAttribute(self, obj, projection, flag):
        """Applies projection.attribute.transform in projection if any to obj.

    Args:
      obj: An object.
      projection: Projection _Tree node.
      flag: A bitmask of DEFAULT, INNER, PROJECT.

    Returns:
      The transformed obj if there was a transform, otherwise the original obj.
    """
        if flag < self._projection.PROJECT:
            return None
        if projection and projection.attribute and projection.attribute.transform and self._TransformIsEnabled(projection.attribute.transform):
            return projection.attribute.transform.Evaluate(obj)
        return self._Project(obj, projection, flag, leaf=True)

    def _ProjectClass(self, obj, projection, flag):
        """Converts class object to a dict.

    Private and callable attributes are omitted in the dict.

    Args:
      obj: The class object to convert.
      projection: Projection _Tree node.
      flag: A bitmask of DEFAULT, INNER, PROJECT.

    Returns:
      The dict representing the class object.
    """
        r = {}
        exclude = set()
        if isinstance(obj, datetime.datetime):
            r['datetime'] = six.text_type(obj)
            exclude.update(('max', 'min', 'resolution', 'tzinfo'))
        else:
            try:
                exclude.update([a for a in dir(obj.__class__) if a.isupper()])
            except AttributeError:
                pass
        for attr in dir(obj):
            if attr.startswith('_'):
                continue
            if attr in exclude:
                continue
            try:
                value = getattr(obj, attr)
            except:
                continue
            if hasattr(value, '__call__'):
                continue
            f = flag
            if attr in projection.tree:
                child_projection = projection.tree[attr]
                f |= child_projection.attribute.flag
                if f < self._projection.INNER:
                    continue
                r[attr] = self._Project(value, child_projection, f)
            else:
                r[attr] = self._ProjectAttribute(value, self._projection.GetEmpty(), f)
        return r

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

    def SetByColumns(self, enable):
        """Sets the projection to list-of-columns mode.

    Args:
      enable: Enables projection to a list-of-columns if True.
    """
        self._by_columns = enable

    def SetIgnoreDefaultTransforms(self, enable):
        """Sets the ignore default transforms mode.

    Args:
      enable: Disable default projection transforms if True.
    """
        self._ignore_default_transforms = enable

    def SetRetainNoneValues(self, enable):
        """Sets the projection to retain-none-values mode.

    Args:
      enable: Enables projection to a retain-none-values if True.
    """
        self._retain_none_values = enable

    def Evaluate(self, obj):
        """Serializes/projects/transforms obj.

    A default or empty projection expression simply converts a resource object
    to a JSON-serializable copy of the object.

    Args:
      obj: An object.

    Returns:
      A JSON-serializeable object containing only the key values selected by
        the projection. The return value is a deep copy of the object: changes
        to the input object do not affect the JSON-serializable copy.
    """
        self._transforms_enabled = self._transforms_enabled_attribute
        if not self._by_columns or not self._columns:
            if self._columns:
                self._retain_none_values = False
                flag = self._projection.DEFAULT
            else:
                flag = self._projection.PROJECT
            if hasattr(obj, 'MakeSerializable'):
                obj = obj.MakeSerializable()
            return self._Project(obj, self._projection.Tree(), flag)
        obj_serialized = self._Project(obj, self._projection.GetEmpty(), self._projection.PROJECT)
        if self._transforms_enabled_attribute is None:
            self._transforms_enabled = not self._ignore_default_transforms
        columns = []
        for column in self._columns:
            val = resource_property.Get(obj_serialized, column.key) if column.key else obj_serialized
            if column.attribute.transform and self._TransformIsEnabled(column.attribute.transform):
                val = column.attribute.transform.Evaluate(val, obj)
            columns.append(val)
        return columns

    def Projection(self):
        """Returns the ProjectionSpec object for the projector.

    Returns:
      The ProjectionSpec object for the projector.
    """
        return self._projection