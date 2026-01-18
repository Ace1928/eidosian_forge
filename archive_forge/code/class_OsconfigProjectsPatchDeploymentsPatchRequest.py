from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class OsconfigProjectsPatchDeploymentsPatchRequest(_messages.Message):
    """A OsconfigProjectsPatchDeploymentsPatchRequest object.

  Fields:
    name: Unique name for the patch deployment resource in a project. The
      patch deployment name is in the form:
      `projects/{project_id}/patchDeployments/{patch_deployment_id}`. This
      field is ignored when you create a new patch deployment.
    patchDeployment: A PatchDeployment resource to be passed as the request
      body.
    updateMask: Optional. Field mask that controls which fields of the patch
      deployment should be updated.
  """
    name = _messages.StringField(1, required=True)
    patchDeployment = _messages.MessageField('PatchDeployment', 2)
    updateMask = _messages.StringField(3)