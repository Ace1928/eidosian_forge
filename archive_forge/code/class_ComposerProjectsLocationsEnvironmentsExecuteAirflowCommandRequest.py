from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComposerProjectsLocationsEnvironmentsExecuteAirflowCommandRequest(_messages.Message):
    """A ComposerProjectsLocationsEnvironmentsExecuteAirflowCommandRequest
  object.

  Fields:
    environment: The resource name of the environment in the form: "projects/{
      projectId}/locations/{locationId}/environments/{environmentId}".
    executeAirflowCommandRequest: A ExecuteAirflowCommandRequest resource to
      be passed as the request body.
  """
    environment = _messages.StringField(1, required=True)
    executeAirflowCommandRequest = _messages.MessageField('ExecuteAirflowCommandRequest', 2)