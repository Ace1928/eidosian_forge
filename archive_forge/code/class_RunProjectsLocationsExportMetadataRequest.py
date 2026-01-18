from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RunProjectsLocationsExportMetadataRequest(_messages.Message):
    """A RunProjectsLocationsExportMetadataRequest object.

  Fields:
    name: Required. The name of the resource of which metadata should be
      exported. Format: `projects/{project_id_or_number}/locations/{location}/
      services/{service}` for Service `projects/{project_id_or_number}/locatio
      ns/{location}/services/{service}/revisions/{revision}` for Revision `pro
      jects/{project_id_or_number}/locations/{location}/jobs/{job}/executions/
      {execution}` for Execution
  """
    name = _messages.StringField(1, required=True)