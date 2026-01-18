from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClouddeployProjectsLocationsDeliveryPipelinesAutomationRunsListRequest(_messages.Message):
    """A ClouddeployProjectsLocationsDeliveryPipelinesAutomationRunsListRequest
  object.

  Fields:
    filter: Filter automationRuns to be returned. All fields can be used in
      the filter.
    orderBy: Field to sort by.
    pageSize: The maximum number of automationRuns to return. The service may
      return fewer than this value. If unspecified, at most 50 automationRuns
      will be returned. The maximum value is 1000; values above 1000 will be
      set to 1000.
    pageToken: A page token, received from a previous `ListAutomationRuns`
      call. Provide this to retrieve the subsequent page. When paginating, all
      other provided parameters match the call that provided the page token.
    parent: Required. The parent `Delivery Pipeline`, which owns this
      collection of automationRuns. Format must be `projects/{project}/locatio
      ns/{location}/deliveryPipelines/{delivery_pipeline}`.
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)