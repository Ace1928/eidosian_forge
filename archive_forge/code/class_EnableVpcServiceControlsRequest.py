from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EnableVpcServiceControlsRequest(_messages.Message):
    """Request to enable VPC service controls.

  Fields:
    consumerNetwork: Required. The network that the consumer is using to
      connect with services. Must be in the form of
      projects/{project}/global/networks/{network} {project} is a project
      number, as in '12345' {network} is network name.
  """
    consumerNetwork = _messages.StringField(1)