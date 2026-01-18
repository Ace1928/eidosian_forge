from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConnectorsProjectsLocationsProvidersGetRequest(_messages.Message):
    """A ConnectorsProjectsLocationsProvidersGetRequest object.

  Fields:
    name: Required. Resource name of the form:
      `projects/*/locations/*/providers/*` Only global location is supported
      for Provider resource.
  """
    name = _messages.StringField(1, required=True)