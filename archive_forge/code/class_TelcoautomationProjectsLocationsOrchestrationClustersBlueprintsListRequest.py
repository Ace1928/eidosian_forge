from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TelcoautomationProjectsLocationsOrchestrationClustersBlueprintsListRequest(_messages.Message):
    """A
  TelcoautomationProjectsLocationsOrchestrationClustersBlueprintsListRequest
  object.

  Fields:
    filter: Optional. Filtering only supports equality on blueprint state. It
      should be in the form: "state = DRAFT". `OR` operator can be used to get
      response for multiple states. e.g. "state = DRAFT OR state = PROPOSED".
    pageSize: Optional. The maximum number of blueprints to return per page.
    pageToken: Optional. The page token, received from a previous
      ListBlueprints call. It can be provided to retrieve the subsequent page.
    parent: Required. The name of parent orchestration cluster resource.
      Format should be - "projects/{project_id}/locations/{location_name}/orch
      estrationClusters/{orchestration_cluster}".
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)