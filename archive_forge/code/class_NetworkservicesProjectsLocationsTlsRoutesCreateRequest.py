from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsTlsRoutesCreateRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsTlsRoutesCreateRequest object.

  Fields:
    parent: Required. The parent resource of the TlsRoute. Must be in the
      format `projects/*/locations/global`.
    tlsRoute: A TlsRoute resource to be passed as the request body.
    tlsRouteId: Required. Short name of the TlsRoute resource to be created.
  """
    parent = _messages.StringField(1, required=True)
    tlsRoute = _messages.MessageField('TlsRoute', 2)
    tlsRouteId = _messages.StringField(3)