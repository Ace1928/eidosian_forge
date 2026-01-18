from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComposerProjectsLocationsEnvironmentsListImportErrorsRequest(_messages.Message):
    """A ComposerProjectsLocationsEnvironmentsListImportErrorsRequest object.

  Fields:
    environment: Required. List import errors in the given Composer
      environment. Environment name must be in the form: "projects/{projectId}
      /locations/{locationId}/environments/{environmentId}".
    pageSize: The maximum number of DAGs to return.
    pageToken: The next_page_token returned from a previous List request.
  """
    environment = _messages.StringField(1, required=True)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)