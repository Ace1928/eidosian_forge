from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsContaineranalysisV1alpha1OperationMetadata(_messages.Message):
    """Metadata for all operations used and required for all operations that
  created by Container Analysis Providers

  Fields:
    createTime: Output only. The time this operation was created.
    endTime: Output only. The time that this operation was marked completed or
      failed.
  """
    createTime = _messages.StringField(1)
    endTime = _messages.StringField(2)