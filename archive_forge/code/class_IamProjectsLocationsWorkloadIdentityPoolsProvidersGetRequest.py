from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamProjectsLocationsWorkloadIdentityPoolsProvidersGetRequest(_messages.Message):
    """A IamProjectsLocationsWorkloadIdentityPoolsProvidersGetRequest object.

  Fields:
    name: Required. The name of the provider to retrieve.
  """
    name = _messages.StringField(1, required=True)