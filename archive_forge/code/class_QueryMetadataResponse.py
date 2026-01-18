from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QueryMetadataResponse(_messages.Message):
    """Response message for DataprocMetastore.QueryMetadata.

  Fields:
    resultManifestUri: The manifest URI is link to a JSON instance in Cloud
      Storage. This instance manifests immediately along with
      QueryMetadataResponse. The content of the URI is not retriable until the
      long-running operation query against the metadata finishes.
  """
    resultManifestUri = _messages.StringField(1)