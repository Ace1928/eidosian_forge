from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class JSONSchemaPropsOrArray(_messages.Message):
    """JSONSchemaPropsOrArray represents a value that can either be a
  JSONSchemaProps or an array of JSONSchemaProps. Mainly here for
  serialization purposes.

  Fields:
    jsonSchemas: A JSONSchemaProps attribute.
    schema: A JSONSchemaProps attribute.
  """
    jsonSchemas = _messages.MessageField('JSONSchemaProps', 1, repeated=True)
    schema = _messages.MessageField('JSONSchemaProps', 2)