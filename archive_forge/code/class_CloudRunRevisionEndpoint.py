from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudRunRevisionEndpoint(_messages.Message):
    """Wrapper for Cloud Run revision attributes.

  Fields:
    uri: A [Cloud Run](https://cloud.google.com/run) [revision](https://cloud.
      google.com/run/docs/reference/rest/v1/namespaces.revisions/get) URI. The
      format is: projects/{project}/locations/{location}/revisions/{revision}
  """
    uri = _messages.StringField(1)