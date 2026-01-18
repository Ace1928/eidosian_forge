from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IapProjectsIapTunnelLocationsDestGroupsCreateRequest(_messages.Message):
    """A IapProjectsIapTunnelLocationsDestGroupsCreateRequest object.

  Fields:
    parent: Required. Google Cloud Project ID and location. In the following
      format: `projects/{project_number/id}/iap_tunnel/locations/{location}`.
    tunnelDestGroup: A TunnelDestGroup resource to be passed as the request
      body.
    tunnelDestGroupId: Required. The ID to use for the TunnelDestGroup, which
      becomes the final component of the resource name. This value must be
      4-63 characters, and valid characters are `[a-z]-`.
  """
    parent = _messages.StringField(1, required=True)
    tunnelDestGroup = _messages.MessageField('TunnelDestGroup', 2)
    tunnelDestGroupId = _messages.StringField(3)