from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class OsconfigProjectsPatchDeploymentsGetRequest(_messages.Message):
    """A OsconfigProjectsPatchDeploymentsGetRequest object.

  Fields:
    name: Required. The resource name of the patch deployment in the form
      `projects/*/patchDeployments/*`.
  """
    name = _messages.StringField(1, required=True)