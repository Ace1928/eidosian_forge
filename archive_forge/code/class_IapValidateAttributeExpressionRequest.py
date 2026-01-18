from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IapValidateAttributeExpressionRequest(_messages.Message):
    """A IapValidateAttributeExpressionRequest object.

  Fields:
    expression: Required. User input string expression. Should be of the form
      `attributes.saml_attributes.filter(attribute, attribute.name in
      ['{attribute_name}', '{attribute_name}'])`
    name: Required. The resource name of the IAP protected resource.
  """
    expression = _messages.StringField(1)
    name = _messages.StringField(2, required=True)