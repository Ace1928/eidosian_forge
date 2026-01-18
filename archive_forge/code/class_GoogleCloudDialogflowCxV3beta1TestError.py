from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3beta1TestError(_messages.Message):
    """Error info for running a test.

  Fields:
    status: The status associated with the test.
    testCase: The test case resource name.
    testTime: The timestamp when the test was completed.
  """
    status = _messages.MessageField('GoogleRpcStatus', 1)
    testCase = _messages.StringField(2)
    testTime = _messages.StringField(3)