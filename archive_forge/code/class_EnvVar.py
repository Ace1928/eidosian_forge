from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EnvVar(_messages.Message):
    """EnvVar represents an environment variable present in a Container.

  Fields:
    name: Required. Name of the environment variable.
    value: Value of the environment variable. Defaults to "". Variable
      references are not supported in Cloud Run.
    valueFrom: Source for the environment variable's value. Only supports
      secret_key_ref. Cannot be used if value is not empty.
  """
    name = _messages.StringField(1)
    value = _messages.StringField(2)
    valueFrom = _messages.MessageField('EnvVarSource', 3)