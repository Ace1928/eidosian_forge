from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildProjectsLocationsConnectionsRepositoriesGetRequest(_messages.Message):
    """A CloudbuildProjectsLocationsConnectionsRepositoriesGetRequest object.

  Fields:
    name: Required. The name of the Repository to retrieve. Format:
      `projects/*/locations/*/connections/*/repositories/*`.
  """
    name = _messages.StringField(1, required=True)