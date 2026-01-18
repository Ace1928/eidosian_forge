from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DeploymentmanagerTypesGetRequest(_messages.Message):
    """A DeploymentmanagerTypesGetRequest object.

  Fields:
    project: The project ID for this request.
    type: The name of the type for this request.
  """
    project = _messages.StringField(1, required=True)
    type = _messages.StringField(2, required=True)