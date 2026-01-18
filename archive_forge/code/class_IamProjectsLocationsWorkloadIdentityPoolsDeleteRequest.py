from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamProjectsLocationsWorkloadIdentityPoolsDeleteRequest(_messages.Message):
    """A IamProjectsLocationsWorkloadIdentityPoolsDeleteRequest object.

  Fields:
    name: Required. The name of the pool to delete.
  """
    name = _messages.StringField(1, required=True)