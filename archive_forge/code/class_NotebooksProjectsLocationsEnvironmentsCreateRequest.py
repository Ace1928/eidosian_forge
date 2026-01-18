from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NotebooksProjectsLocationsEnvironmentsCreateRequest(_messages.Message):
    """A NotebooksProjectsLocationsEnvironmentsCreateRequest object.

  Fields:
    environment: A Environment resource to be passed as the request body.
    environmentId: Required. User-defined unique ID of this environment. The
      `environment_id` must be 1 to 63 characters long and contain only
      lowercase letters, numeric characters, and dashes. The first character
      must be a lowercase letter and the last character cannot be a dash.
    parent: Required. Format: `projects/{project_id}/locations/{location}`
  """
    environment = _messages.MessageField('Environment', 1)
    environmentId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)