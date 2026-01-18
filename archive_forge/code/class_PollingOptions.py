from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PollingOptions(_messages.Message):
    """A PollingOptions object.

  Fields:
    diagnostics: An array of diagnostics to be collected by Deployment
      Manager, these diagnostics will be displayed to the user.
    failCondition: JsonPath expression that determines if the request failed.
    finishCondition: JsonPath expression that determines if the request is
      completed.
    pollingLink: JsonPath expression that evaluates to string, it indicates
      where to poll.
    targetLink: JsonPath expression, after polling is completed, indicates
      where to fetch the resource.
  """
    diagnostics = _messages.MessageField('Diagnostic', 1, repeated=True)
    failCondition = _messages.StringField(2)
    finishCondition = _messages.StringField(3)
    pollingLink = _messages.StringField(4)
    targetLink = _messages.StringField(5)