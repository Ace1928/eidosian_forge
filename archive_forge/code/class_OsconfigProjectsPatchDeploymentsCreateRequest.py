from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class OsconfigProjectsPatchDeploymentsCreateRequest(_messages.Message):
    """A OsconfigProjectsPatchDeploymentsCreateRequest object.

  Fields:
    parent: Required. The project to apply this patch deployment to in the
      form `projects/*`.
    patchDeployment: A PatchDeployment resource to be passed as the request
      body.
    patchDeploymentId: Required. A name for the patch deployment in the
      project. When creating a name the following rules apply: * Must contain
      only lowercase letters, numbers, and hyphens. * Must start with a
      letter. * Must be between 1-63 characters. * Must end with a number or a
      letter. * Must be unique within the project.
  """
    parent = _messages.StringField(1, required=True)
    patchDeployment = _messages.MessageField('PatchDeployment', 2)
    patchDeploymentId = _messages.StringField(3)