from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1EnvVar(_messages.Message):
    """Represents an environment variable present in a Container or Python
  Module.

  Fields:
    name: Required. Name of the environment variable. Must be a valid C
      identifier.
    value: Required. Variables that reference a $(VAR_NAME) are expanded using
      the previous defined environment variables in the container and any
      service environment variables. If a variable cannot be resolved, the
      reference in the input string will be unchanged. The $(VAR_NAME) syntax
      can be escaped with a double $$, ie: $$(VAR_NAME). Escaped references
      will never be expanded, regardless of whether the variable exists or
      not.
  """
    name = _messages.StringField(1)
    value = _messages.StringField(2)