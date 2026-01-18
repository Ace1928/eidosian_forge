from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GsuiteaddonsProjectsDeploymentsGetInstallStatusRequest(_messages.Message):
    """A GsuiteaddonsProjectsDeploymentsGetInstallStatusRequest object.

  Fields:
    name: Required. The full resource name of the deployment. Example:
      `projects/my_project/deployments/my_deployment/installStatus`.
  """
    name = _messages.StringField(1, required=True)