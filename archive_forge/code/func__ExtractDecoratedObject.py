from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import six
def _ExtractDecoratedObject(proto):
    """Extracts an object from the proto representation of the JSON object.

  Args:
    proto: A protocol representation of a JSON object.
  Returns:
    A clean representation of the JSON object. If it was an object
    representing a primitive, then that primitive.
  """
    prop_dict = {}
    for prop in proto.object_value.properties:
        prop_dict[prop.key] = prop.value
    ty = prop_dict.get('@type', None)
    retriever = ty and _VALUE_RETRIEVERS.get(ty.string_value, None)
    if not ty or not retriever:
        return dict(((k, _ExtractValue(v)) for k, v in six.iteritems(prop_dict)))
    try:
        return retriever(prop_dict['value'])
    except KeyError:
        return 'Missing value for type [{0}] in proto [{1}]'.format(ty.string_value, proto)