from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConfigProjectsLocationsDeploymentsImportStateRequest(_messages.Message):
    """A ConfigProjectsLocationsDeploymentsImportStateRequest object.

  Fields:
    importStatefileRequest: A ImportStatefileRequest resource to be passed as
      the request body.
    parent: Required. The parent in whose context the statefile is listed. The
      parent value is in the format:
      'projects/{project_id}/locations/{location}/deployments/{deployment}'.
  """
    importStatefileRequest = _messages.MessageField('ImportStatefileRequest', 1)
    parent = _messages.StringField(2, required=True)