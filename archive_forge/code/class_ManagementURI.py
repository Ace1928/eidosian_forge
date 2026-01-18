from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ManagementURI(_messages.Message):
    """ManagementURI for the Management Server resource.

  Fields:
    api: Output only. The ManagementServer AGM/RD API URL.
    webUi: Output only. The ManagementServer AGM/RD WebUI URL.
  """
    api = _messages.StringField(1)
    webUi = _messages.StringField(2)