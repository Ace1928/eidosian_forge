from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EdgenetworkProjectsLocationsZonesInterconnectsDiagnoseRequest(_messages.Message):
    """A EdgenetworkProjectsLocationsZonesInterconnectsDiagnoseRequest object.

  Fields:
    name: Required. The name of the interconnect resource.
  """
    name = _messages.StringField(1, required=True)