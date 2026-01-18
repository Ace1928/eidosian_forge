from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServicenetworkingServicesConnectionsPatchRequest(_messages.Message):
    """A ServicenetworkingServicesConnectionsPatchRequest object.

  Fields:
    connection: A Connection resource to be passed as the request body.
    force: If a previously defined allocated range is removed, force flag must
      be set to true.
    name: The private service connection that connects to a service producer
      organization. The name includes both the private service name and the
      VPC network peering name in the format of
      `services/{peering_service_name}/connections/{vpc_peering_name}`. For
      Google services that support this functionality, this is `services/servi
      cenetworking.googleapis.com/connections/servicenetworking-googleapis-
      com`.
    updateMask: The update mask. If this is omitted, it defaults to "*". You
      can only update the listed peering ranges.
  """
    connection = _messages.MessageField('Connection', 1)
    force = _messages.BooleanField(2)
    name = _messages.StringField(3, required=True)
    updateMask = _messages.StringField(4)