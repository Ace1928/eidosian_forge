from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PolicyanalyzerProjectsLocationsActivityTypesActivitiesQueryRequest(_messages.Message):
    """A PolicyanalyzerProjectsLocationsActivityTypesActivitiesQueryRequest
  object.

  Fields:
    filter: Optional. Filter expression to restrict the activities returned.
      For serviceAccountLastAuthentication activities, supported filters are:
      - `activities.full_resource_name {=} [STRING]` -
      `activities.fullResourceName {=} [STRING]` where `[STRING]` is the full
      resource name of the service account. For
      serviceAccountKeyLastAuthentication activities, supported filters are: -
      `activities.full_resource_name {=} [STRING]` -
      `activities.fullResourceName {=} [STRING]` where `[STRING]` is the full
      resource name of the service account key.
    pageSize: Optional. The maximum number of results to return from this
      request. Max limit is 1000. Non-positive values are ignored. The
      presence of `nextPageToken` in the response indicates that more results
      might be available.
    pageToken: Optional. If present, then retrieve the next batch of results
      from the preceding call to this method. `pageToken` must be the value of
      `nextPageToken` from the previous response. The values of other method
      parameters should be identical to those in the previous call.
    parent: Required. The container resource on which to execute the request.
      Acceptable formats: `projects/[PROJECT_ID|PROJECT_NUMBER]/locations/[LOC
      ATION]/activityTypes/[ACTIVITY_TYPE]` LOCATION here refers to Google
      Cloud Locations: https://cloud.google.com/about/locations/
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)