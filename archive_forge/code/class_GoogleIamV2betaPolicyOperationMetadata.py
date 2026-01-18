from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleIamV2betaPolicyOperationMetadata(_messages.Message):
    """Metadata for long-running Policy operations.

  Fields:
    createTime: Timestamp when the google.longrunning.Operation was created.
  """
    createTime = _messages.StringField(1)