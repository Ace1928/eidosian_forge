from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ValueInfo(_messages.Message):
    """Annotatated property value.

  Fields:
    annotation: Annotation, comment or explanation why the property was set.
    overriddenValue: Optional. Value which was replaced by the corresponding
      component.
    value: Property value.
  """
    annotation = _messages.StringField(1)
    overriddenValue = _messages.StringField(2)
    value = _messages.StringField(3)