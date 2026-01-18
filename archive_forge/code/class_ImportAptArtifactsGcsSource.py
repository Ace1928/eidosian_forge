from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ImportAptArtifactsGcsSource(_messages.Message):
    """Google Cloud Storage location where the artifacts currently reside.

  Fields:
    uris: Cloud Storage paths URI (e.g., gs://my_bucket//my_object).
    useWildcards: Supports URI wildcards for matching multiple objects from a
      single URI.
  """
    uris = _messages.StringField(1, repeated=True)
    useWildcards = _messages.BooleanField(2)