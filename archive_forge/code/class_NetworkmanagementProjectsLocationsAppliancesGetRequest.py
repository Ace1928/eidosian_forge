from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkmanagementProjectsLocationsAppliancesGetRequest(_messages.Message):
    """A NetworkmanagementProjectsLocationsAppliancesGetRequest object.

  Fields:
    name: Required. A name of the Appliance to get. Must be in the format
      `projects/{project}/locations/{location}/appliances/{appliance}`.
  """
    name = _messages.StringField(1, required=True)