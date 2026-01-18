from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import six
def _ExtractStep(step_msg):
    """Converts a Step message into a dict with more sensible structure.

  Args:
    step_msg: A Step message.
  Returns:
    A dict with the cleaned up information.
  """
    properties = {}
    if step_msg.properties:
        for prop in step_msg.properties.additionalProperties:
            if prop.key not in _EXCLUDED_PROPERTIES:
                properties[prop.key] = _ExtractValue(prop.value)
    return {'kind': step_msg.kind, 'name': step_msg.name, 'properties': properties}