from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeregistryProjectsLocationsApisDeploymentsDeleteRequest(_messages.Message):
    """A ApigeeregistryProjectsLocationsApisDeploymentsDeleteRequest object.

  Fields:
    force: If set to true, any child resources will also be deleted.
      (Otherwise, the request will only work if there are no child resources.)
    name: Required. The name of the deployment to delete. Format:
      `projects/*/locations/*/apis/*/deployments/*`
  """
    force = _messages.BooleanField(1)
    name = _messages.StringField(2, required=True)