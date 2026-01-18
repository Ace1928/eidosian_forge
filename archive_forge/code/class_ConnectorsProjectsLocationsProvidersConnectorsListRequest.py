from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConnectorsProjectsLocationsProvidersConnectorsListRequest(_messages.Message):
    """A ConnectorsProjectsLocationsProvidersConnectorsListRequest object.

  Fields:
    filter: Filter string.
    pageSize: Page size.
    pageToken: Page token.
    parent: Required. Parent resource of the connectors, of the form:
      `projects/*/locations/*/providers/*` Only global location is supported
      for Connector resource.
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)