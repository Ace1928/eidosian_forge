from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkmanagementProjectsLocationsDeploymentsDeleteRequest(_messages.Message):
    """A NetworkmanagementProjectsLocationsDeploymentsDeleteRequest object.

  Fields:
    name: Required. name of the deployment to be deleted
  """
    name = _messages.StringField(1, required=True)