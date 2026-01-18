from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsTcpRoutesDeleteRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsTcpRoutesDeleteRequest object.

  Fields:
    name: Required. A name of the TcpRoute to delete. Must be in the format
      `projects/*/locations/global/tcpRoutes/*`.
  """
    name = _messages.StringField(1, required=True)