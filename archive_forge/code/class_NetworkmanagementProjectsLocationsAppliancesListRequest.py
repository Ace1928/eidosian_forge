from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkmanagementProjectsLocationsAppliancesListRequest(_messages.Message):
    """A NetworkmanagementProjectsLocationsAppliancesListRequest object.

  Fields:
    pageSize: Maximum number of Appliances to return per call.
    pageToken: The value returned by the last `ListAppliancesResponse`
      Indicates that this is a continuation of a prior `ListAppliances` call,
      and that the system should return the next page of data.
    parent: Required. The project and location from which the Appliance should
      be listed, specified in the format
      `projects/{project}/locations/{location}`.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)