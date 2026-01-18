from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmmigrationProjectsLocationsSourcesDatacenterConnectorsGetRequest(_messages.Message):
    """A VmmigrationProjectsLocationsSourcesDatacenterConnectorsGetRequest
  object.

  Fields:
    name: Required. The name of the DatacenterConnector.
  """
    name = _messages.StringField(1, required=True)