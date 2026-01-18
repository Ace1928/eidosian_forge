from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IapProjectsIapTunnelLocationsDestGroupsGetRequest(_messages.Message):
    """A IapProjectsIapTunnelLocationsDestGroupsGetRequest object.

  Fields:
    name: Required. Name of the TunnelDestGroup to be fetched. In the
      following format: `projects/{project_number/id}/iap_tunnel/locations/{lo
      cation}/destGroups/{dest_group}`.
  """
    name = _messages.StringField(1, required=True)