from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AssuredworkloadsOrganizationsLocationsWorkloadsCreateRequest(_messages.Message):
    """A AssuredworkloadsOrganizationsLocationsWorkloadsCreateRequest object.

  Fields:
    externalId: Optional. A identifier associated with the workload and
      underlying projects which allows for the break down of billing costs for
      a workload. The value provided for the identifier will add a label to
      the workload and contained projects with the identifier as the value.
    googleCloudAssuredworkloadsV1Workload: A
      GoogleCloudAssuredworkloadsV1Workload resource to be passed as the
      request body.
    parent: Required. The resource name of the new Workload's parent. Must be
      of the form `organizations/{org_id}/locations/{location_id}`.
  """
    externalId = _messages.StringField(1)
    googleCloudAssuredworkloadsV1Workload = _messages.MessageField('GoogleCloudAssuredworkloadsV1Workload', 2)
    parent = _messages.StringField(3, required=True)