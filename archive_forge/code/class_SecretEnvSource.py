from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecretEnvSource(_messages.Message):
    """Not supported by Cloud Run. SecretEnvSource selects a Secret to populate
  the environment variables with. The contents of the target Secret's Data
  field will represent the key-value pairs as environment variables.

  Fields:
    localObjectReference: This field should not be used directly as it is
      meant to be inlined directly into the message. Use the "name" field
      instead.
    name: The Secret to select from.
    optional: Specify whether the Secret must be defined
  """
    localObjectReference = _messages.MessageField('LocalObjectReference', 1)
    name = _messages.StringField(2)
    optional = _messages.BooleanField(3)