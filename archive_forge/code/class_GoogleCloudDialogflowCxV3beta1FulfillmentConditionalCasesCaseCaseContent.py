from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3beta1FulfillmentConditionalCasesCaseCaseContent(_messages.Message):
    """The list of messages or conditional cases to activate for this case.

  Fields:
    additionalCases: Additional cases to be evaluated.
    message: Returned message.
  """
    additionalCases = _messages.MessageField('GoogleCloudDialogflowCxV3beta1FulfillmentConditionalCases', 1)
    message = _messages.MessageField('GoogleCloudDialogflowCxV3beta1ResponseMessage', 2)