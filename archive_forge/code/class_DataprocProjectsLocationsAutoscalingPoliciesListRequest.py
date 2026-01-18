from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataprocProjectsLocationsAutoscalingPoliciesListRequest(_messages.Message):
    """A DataprocProjectsLocationsAutoscalingPoliciesListRequest object.

  Fields:
    pageSize: Optional. The maximum number of results to return in each
      response. Must be less than or equal to 1000. Defaults to 100.
    pageToken: Optional. The page token, returned by a previous call, to
      request the next page of results.
    parent: Required. The "resource name" of the region or location, as
      described in https://cloud.google.com/apis/design/resource_names. For
      projects.regions.autoscalingPolicies.list, the resource name of the
      region has the following format: projects/{project_id}/regions/{region}
      For projects.locations.autoscalingPolicies.list, the resource name of
      the location has the following format:
      projects/{project_id}/locations/{location}
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)