from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NotebooksProjectsLocationsRuntimesMigrateRequest(_messages.Message):
    """A NotebooksProjectsLocationsRuntimesMigrateRequest object.

  Fields:
    migrateRuntimeRequest: A MigrateRuntimeRequest resource to be passed as
      the request body.
    name: Required. Format:
      `projects/{project_id}/locations/{location}/runtimes/{runtime_id}`
  """
    migrateRuntimeRequest = _messages.MessageField('MigrateRuntimeRequest', 1)
    name = _messages.StringField(2, required=True)