from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContainerAnalysisStorage(_messages.Message):
    """Configuration for provenance storage in Container Analysis.

  Fields:
    project: The GCP project that stores provenance. Format:
      `projects/{project_id}` or `projects/{project_number}`
  """
    project = _messages.StringField(1)