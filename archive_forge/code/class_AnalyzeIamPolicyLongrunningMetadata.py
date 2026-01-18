from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AnalyzeIamPolicyLongrunningMetadata(_messages.Message):
    """Represents the metadata of the longrunning operation for the
  AnalyzeIamPolicyLongrunning RPC.

  Fields:
    createTime: Output only. The time the operation was created.
  """
    createTime = _messages.StringField(1)