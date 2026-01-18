from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TelcoautomationProjectsLocationsOrchestrationClustersBlueprintsListRevisionsRequest(_messages.Message):
    """A TelcoautomationProjectsLocationsOrchestrationClustersBlueprintsListRev
  isionsRequest object.

  Fields:
    name: Required. The name of the blueprint to list revisions for.
    pageSize: The maximum number of revisions to return per page.
    pageToken: The page token, received from a previous ListBlueprintRevisions
      call It can be provided to retrieve the subsequent page.
  """
    name = _messages.StringField(1, required=True)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)