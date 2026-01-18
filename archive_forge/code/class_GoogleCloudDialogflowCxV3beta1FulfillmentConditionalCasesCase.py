from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3beta1FulfillmentConditionalCasesCase(_messages.Message):
    """Each case has a Boolean condition. When it is evaluated to be True, the
  corresponding messages will be selected and evaluated recursively.

  Fields:
    caseContent: A list of case content.
    condition: The condition to activate and select this case. Empty means the
      condition is always true. The condition is evaluated against form
      parameters or session parameters. See the [conditions reference](https:/
      /cloud.google.com/dialogflow/cx/docs/reference/condition).
  """
    caseContent = _messages.MessageField('GoogleCloudDialogflowCxV3beta1FulfillmentConditionalCasesCaseCaseContent', 1, repeated=True)
    condition = _messages.StringField(2)