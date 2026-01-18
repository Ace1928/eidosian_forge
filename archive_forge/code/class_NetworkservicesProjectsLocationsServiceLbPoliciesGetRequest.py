from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsServiceLbPoliciesGetRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsServiceLbPoliciesGetRequest object.

  Fields:
    name: Required. A name of the ServiceLbPolicy to get. Must be in the
      format `projects/{project}/locations/{location}/serviceLbPolicies/*`.
  """
    name = _messages.StringField(1, required=True)