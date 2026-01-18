from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeRegionInstanceGroupManagersCreateInstancesRequest(_messages.Message):
    """A ComputeRegionInstanceGroupManagersCreateInstancesRequest object.

  Fields:
    instanceGroupManager: The name of the managed instance group. It should
      conform to RFC1035.
    project: Project ID for this request.
    region: The name of the region where the managed instance group is
      located. It should conform to RFC1035.
    regionInstanceGroupManagersCreateInstancesRequest: A
      RegionInstanceGroupManagersCreateInstancesRequest resource to be passed
      as the request body.
    requestId: An optional request ID to identify requests. Specify a unique
      request ID so that if you must retry your request, the server will know
      to ignore the request if it has already been completed. For example,
      consider a situation where you make an initial request and the request
      times out. If you make the request again with the same request ID, the
      server can check if original operation with the same request ID was
      received, and if so, will ignore the second request. The request ID must
      be a valid UUID with the exception that zero UUID is not supported (
      00000000-0000-0000-0000-000000000000).
  """
    instanceGroupManager = _messages.StringField(1, required=True)
    project = _messages.StringField(2, required=True)
    region = _messages.StringField(3, required=True)
    regionInstanceGroupManagersCreateInstancesRequest = _messages.MessageField('RegionInstanceGroupManagersCreateInstancesRequest', 4)
    requestId = _messages.StringField(5)