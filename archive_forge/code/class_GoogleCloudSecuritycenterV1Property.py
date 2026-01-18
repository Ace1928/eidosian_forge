from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSecuritycenterV1Property(_messages.Message):
    """An individual name-value pair that defines a custom source property.

  Fields:
    name: Name of the property for the custom output.
    valueExpression: The CEL expression for the custom output. A resource
      property can be specified to return the value of the property or a text
      string enclosed in quotation marks.
  """
    name = _messages.StringField(1)
    valueExpression = _messages.MessageField('Expr', 2)