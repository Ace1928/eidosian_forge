from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AssuredworkloadsOrganizationsLocationsWorkloadsViolationsGetRequest(_messages.Message):
    """A AssuredworkloadsOrganizationsLocationsWorkloadsViolationsGetRequest
  object.

  Fields:
    name: Required. The resource name of the Violation to fetch (ie.
      Violation.name). Format: organizations/{organization}/locations/{locatio
      n}/workloads/{workload}/violations/{violation}
  """
    name = _messages.StringField(1, required=True)