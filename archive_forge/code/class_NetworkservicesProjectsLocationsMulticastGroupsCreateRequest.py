from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsMulticastGroupsCreateRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsMulticastGroupsCreateRequest object.

  Fields:
    multicastGroup: A MulticastGroup resource to be passed as the request
      body.
    multicastGroupId: Required. A unique name for the multicast group. The
      name is restricted to letters, numbers, and hyphen, with the first
      character a letter, and the last a letter or a number. The name must not
      exceed 48 characters.
    parent: Required. The parent resource of the multicast group. Use the
      following format: `projects/*/locations/*`.
    requestId: Optional. An optional request ID to identify requests. Specify
      a unique request ID so that if you must retry your request, the server
      will know to ignore the request if it has already been completed. The
      server will guarantee that for at least 60 minutes after the first
      request. For example, consider a situation where you make an initial
      request and the request times out. If you make the request again with
      the same request ID, the server can check if original operation with the
      same request ID was received, and if so, will ignore the second request.
      This prevents clients from accidentally creating duplicate commitments.
      The request ID must be a valid UUID with the exception that zero UUID is
      not supported (00000000-0000-0000-0000-000000000000).
  """
    multicastGroup = _messages.MessageField('MulticastGroup', 1)
    multicastGroupId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    requestId = _messages.StringField(4)