from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudSqlInstance(_messages.Message):
    """Settings for CloudSQL instance configuration.

  Fields:
    instance: Required. Name of the CloudSQL instance, in the format: ```
      projects/{project}/locations/{location}/instances/{instance} ```
  """
    instance = _messages.StringField(1)