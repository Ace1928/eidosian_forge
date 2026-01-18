from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PipelineResults(_messages.Message):
    """Locations of outputs from kpt pipeline execution.

  Fields:
    artifacts: Location of kpt artifacts in Google Cloud Storage. Format:
      `gs://{bucket}/{object}`
    content: Location of generated manifests in Google Cloud Storage. Format:
      `gs://{bucket}/{object}`
  """
    artifacts = _messages.StringField(1)
    content = _messages.StringField(2)