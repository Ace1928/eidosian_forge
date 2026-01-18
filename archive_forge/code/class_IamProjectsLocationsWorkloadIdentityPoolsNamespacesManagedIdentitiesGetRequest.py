from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamProjectsLocationsWorkloadIdentityPoolsNamespacesManagedIdentitiesGetRequest(_messages.Message):
    """A IamProjectsLocationsWorkloadIdentityPoolsNamespacesManagedIdentitiesGe
  tRequest object.

  Fields:
    name: Required. The name of the managed identity to retrieve.
  """
    name = _messages.StringField(1, required=True)