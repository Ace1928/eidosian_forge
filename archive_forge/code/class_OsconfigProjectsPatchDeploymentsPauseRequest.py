from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class OsconfigProjectsPatchDeploymentsPauseRequest(_messages.Message):
    """A OsconfigProjectsPatchDeploymentsPauseRequest object.

  Fields:
    name: Required. The resource name of the patch deployment in the form
      `projects/*/patchDeployments/*`.
    pausePatchDeploymentRequest: A PausePatchDeploymentRequest resource to be
      passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    pausePatchDeploymentRequest = _messages.MessageField('PausePatchDeploymentRequest', 2)