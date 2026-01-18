from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ToolUseExampleExtensionOperation(_messages.Message):
    """Identifies one operation of the extension.

  Fields:
    extension: Resource name of the extension.
    operationId: Required. Operation ID of the extension.
  """
    extension = _messages.StringField(1)
    operationId = _messages.StringField(2)