from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComposerProjectsLocationsEnvironmentsDagsListRequest(_messages.Message):
    """A ComposerProjectsLocationsEnvironmentsDagsListRequest object.

  Fields:
    pageSize: The maximum number of DAGs to return.
    pageToken: The next_page_token returned from a previous List request.
    parent: Required. List DAGs in the given parent resource. Parent must be
      in the form: "projects/{projectId}/locations/{locationId}/environments/{
      environmentId}".
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)