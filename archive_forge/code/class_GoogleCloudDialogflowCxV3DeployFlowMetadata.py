from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3DeployFlowMetadata(_messages.Message):
    """Metadata returned for the Environments.DeployFlow long running
  operation.

  Fields:
    testErrors: Errors of running deployment tests.
  """
    testErrors = _messages.MessageField('GoogleCloudDialogflowCxV3TestError', 1, repeated=True)