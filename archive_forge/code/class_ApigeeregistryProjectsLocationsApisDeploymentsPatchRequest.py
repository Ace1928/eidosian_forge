from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeregistryProjectsLocationsApisDeploymentsPatchRequest(_messages.Message):
    """A ApigeeregistryProjectsLocationsApisDeploymentsPatchRequest object.

  Fields:
    allowMissing: If set to true, and the deployment is not found, a new
      deployment will be created. In this situation, `update_mask` is ignored.
    apiDeployment: A ApiDeployment resource to be passed as the request body.
    name: Resource name.
    updateMask: The list of fields to be updated. If omitted, all fields are
      updated that are set in the request message (fields set to default
      values are ignored). If an asterisk "*" is specified, all fields are
      updated, including fields that are unspecified/default in the request.
  """
    allowMissing = _messages.BooleanField(1)
    apiDeployment = _messages.MessageField('ApiDeployment', 2)
    name = _messages.StringField(3, required=True)
    updateMask = _messages.StringField(4)