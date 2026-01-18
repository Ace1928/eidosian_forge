from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatafusionProjectsLocationsInstancesDnsPeeringsDeleteRequest(_messages.Message):
    """A DatafusionProjectsLocationsInstancesDnsPeeringsDeleteRequest object.

  Fields:
    name: Required. The name of the DNS peering zone to delete. Format: projec
      ts/{project}/locations/{location}/instances/{instance}/dnsPeerings/{dns_
      peering}
  """
    name = _messages.StringField(1, required=True)