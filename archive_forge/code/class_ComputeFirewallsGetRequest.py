from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeFirewallsGetRequest(_messages.Message):
    """A ComputeFirewallsGetRequest object.

  Fields:
    firewall: Name of the firewall rule to return.
    project: Project ID for this request.
  """
    firewall = _messages.StringField(1, required=True)
    project = _messages.StringField(2, required=True)