from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeregistryProjectsLocationsApisDeploymentsDeleteRevisionRequest(_messages.Message):
    """A ApigeeregistryProjectsLocationsApisDeploymentsDeleteRevisionRequest
  object.

  Fields:
    name: Required. The name of the deployment revision to be deleted, with a
      revision ID explicitly included. Example: `projects/sample/locations/glo
      bal/apis/petstore/deployments/prod@c7cfa2a8`
  """
    name = _messages.StringField(1, required=True)