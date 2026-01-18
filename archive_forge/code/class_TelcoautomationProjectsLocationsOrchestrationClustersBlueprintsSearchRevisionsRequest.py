from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TelcoautomationProjectsLocationsOrchestrationClustersBlueprintsSearchRevisionsRequest(_messages.Message):
    """A TelcoautomationProjectsLocationsOrchestrationClustersBlueprintsSearchR
  evisionsRequest object.

  Fields:
    pageSize: Optional. The maximum number of blueprints revisions to return
      per page. max page size = 100, default page size = 20.
    pageToken: Optional. The page token, received from a previous search call.
      It can be provided to retrieve the subsequent page.
    parent: Required. The name of parent orchestration cluster resource.
      Format should be - "projects/{project_id}/locations/{location_name}/orch
      estrationClusters/{orchestration_cluster}".
    query: Required. Supported queries: 1. "" : Lists all revisions across all
      blueprints. 2. "latest=true" : Lists latest revisions across all
      blueprints. 3. "name={name}" : Lists all revisions of blueprint with
      name {name}. 4. "name={name} latest=true": Lists latest revision of
      blueprint with name {name}
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    query = _messages.StringField(4)