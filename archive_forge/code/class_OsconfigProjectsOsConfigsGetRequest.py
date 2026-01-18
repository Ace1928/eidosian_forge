from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class OsconfigProjectsOsConfigsGetRequest(_messages.Message):
    """A OsconfigProjectsOsConfigsGetRequest object.

  Fields:
    name: The resource name of the OsConfig.
  """
    name = _messages.StringField(1, required=True)