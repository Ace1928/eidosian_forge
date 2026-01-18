from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsTcpRoutesCreateRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsTcpRoutesCreateRequest object.

  Fields:
    parent: Required. The parent resource of the TcpRoute. Must be in the
      format `projects/*/locations/global`.
    tcpRoute: A TcpRoute resource to be passed as the request body.
    tcpRouteId: Required. Short name of the TcpRoute resource to be created.
  """
    parent = _messages.StringField(1, required=True)
    tcpRoute = _messages.MessageField('TcpRoute', 2)
    tcpRouteId = _messages.StringField(3)