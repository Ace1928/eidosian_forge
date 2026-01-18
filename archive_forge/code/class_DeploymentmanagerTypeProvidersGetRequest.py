from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DeploymentmanagerTypeProvidersGetRequest(_messages.Message):
    """A DeploymentmanagerTypeProvidersGetRequest object.

  Fields:
    project: The project ID for this request.
    typeProvider: The name of the type provider for this request.
  """
    project = _messages.StringField(1, required=True)
    typeProvider = _messages.StringField(2, required=True)