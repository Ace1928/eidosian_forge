from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DlpProjectsLocationsConnectionsDeleteRequest(_messages.Message):
    """A DlpProjectsLocationsConnectionsDeleteRequest object.

  Fields:
    name: Required. Resource name of the Connection to be deleted, in the
      format:
      "projects/{project}/locations/{location}/connections/{connection}".
  """
    name = _messages.StringField(1, required=True)