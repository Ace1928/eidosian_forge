from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeregistryProjectsLocationsApisArtifactsGetRequest(_messages.Message):
    """A ApigeeregistryProjectsLocationsApisArtifactsGetRequest object.

  Fields:
    name: Required. The name of the artifact to retrieve. Format:
      `{parent}/artifacts/*`
  """
    name = _messages.StringField(1, required=True)