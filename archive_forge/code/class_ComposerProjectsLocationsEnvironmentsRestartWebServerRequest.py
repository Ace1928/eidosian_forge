from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComposerProjectsLocationsEnvironmentsRestartWebServerRequest(_messages.Message):
    """A ComposerProjectsLocationsEnvironmentsRestartWebServerRequest object.

  Fields:
    name: The resource name of the environment to restart the web server for,
      in the form: "projects/{projectId}/locations/{locationId}/environments/{
      environmentId}"
    restartWebServerRequest: A RestartWebServerRequest resource to be passed
      as the request body.
  """
    name = _messages.StringField(1, required=True)
    restartWebServerRequest = _messages.MessageField('RestartWebServerRequest', 2)