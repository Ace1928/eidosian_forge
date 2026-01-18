from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeForwardingRulesGetRequest(_messages.Message):
    """A ComputeForwardingRulesGetRequest object.

  Fields:
    forwardingRule: Name of the ForwardingRule resource to return.
    project: Project ID for this request.
    region: Name of the region scoping this request.
  """
    forwardingRule = _messages.StringField(1, required=True)
    project = _messages.StringField(2, required=True)
    region = _messages.StringField(3, required=True)