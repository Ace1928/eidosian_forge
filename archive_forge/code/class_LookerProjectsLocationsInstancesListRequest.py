from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LookerProjectsLocationsInstancesListRequest(_messages.Message):
    """A LookerProjectsLocationsInstancesListRequest object.

  Fields:
    pageSize: The maximum number of instances to return. If unspecified at
      most 256 will be returned. The maximum possible value is 2048.
    pageToken: A page token received from a previous ListInstancesRequest.
    parent: Required. Format: `projects/{project}/locations/{location}`.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)