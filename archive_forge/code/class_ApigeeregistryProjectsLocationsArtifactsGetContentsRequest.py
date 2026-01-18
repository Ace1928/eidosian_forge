from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeregistryProjectsLocationsArtifactsGetContentsRequest(_messages.Message):
    """A ApigeeregistryProjectsLocationsArtifactsGetContentsRequest object.

  Fields:
    name: Required. The name of the artifact whose contents should be
      retrieved. Format: `{parent}/artifacts/*`
  """
    name = _messages.StringField(1, required=True)