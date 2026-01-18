from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsMulticastGroupsPatchRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsMulticastGroupsPatchRequest object.

  Fields:
    multicastGroup: A MulticastGroup resource to be passed as the request
      body.
    name: The resource name of the multicast group. Use the following format:
      `projects/*/locations/*/multicastGroups/*`.
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
    updateMask: Required. The field mask specifies the fields to overwrite in
      the multicast group resource by the update. The fields specified in the
      `update_mask` are relative to the resource, not the full request. If a
      field is in the mask, then it is overwritten. If the you do not provide
      a mask, then all fields are overwritten.
  """
    multicastGroup = _messages.MessageField('MulticastGroup', 1)
    name = _messages.StringField(2, required=True)
    requestId = _messages.StringField(3)
    updateMask = _messages.StringField(4)