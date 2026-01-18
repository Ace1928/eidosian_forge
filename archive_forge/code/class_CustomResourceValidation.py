from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class CustomResourceValidation(_messages.Message):
    """CustomResourceValidation is a list of validation methods for
  CustomResources.

  Fields:
    openAPIV3Schema: OpenAPIV3Schema is the OpenAPI v3 schema to be validated
      against.
  """
    openAPIV3Schema = _messages.MessageField('JSONSchemaProps', 1)