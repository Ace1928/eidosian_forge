from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3beta1FulfillmentConditionalCases(_messages.Message):
    """A list of cascading if-else conditions. Cases are mutually exclusive.
  The first one with a matching condition is selected, all the rest ignored.

  Fields:
    cases: A list of cascading if-else conditions.
  """
    cases = _messages.MessageField('GoogleCloudDialogflowCxV3beta1FulfillmentConditionalCasesCase', 1, repeated=True)