from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsLocationsAgentEnvironmentsPatchRequest(_messages.Message):
    """A DialogflowProjectsLocationsAgentEnvironmentsPatchRequest object.

  Fields:
    allowLoadToDraftAndDiscardChanges: Optional. This field is used to prevent
      accidental overwrite of the default environment, which is an operation
      that cannot be undone. To confirm that the caller desires this
      overwrite, this field must be explicitly set to true when updating the
      default environment (environment ID = `-`).
    googleCloudDialogflowV2Environment: A GoogleCloudDialogflowV2Environment
      resource to be passed as the request body.
    name: Output only. The unique identifier of this agent environment.
      Supported formats: - `projects//agent/environments/` -
      `projects//locations//agent/environments/` The environment ID for the
      default environment is `-`.
    updateMask: Required. The mask to control which fields get updated.
  """
    allowLoadToDraftAndDiscardChanges = _messages.BooleanField(1)
    googleCloudDialogflowV2Environment = _messages.MessageField('GoogleCloudDialogflowV2Environment', 2)
    name = _messages.StringField(3, required=True)
    updateMask = _messages.StringField(4)