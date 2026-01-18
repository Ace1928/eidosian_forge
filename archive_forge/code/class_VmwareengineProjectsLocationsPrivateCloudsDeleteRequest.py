from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareengineProjectsLocationsPrivateCloudsDeleteRequest(_messages.Message):
    """A VmwareengineProjectsLocationsPrivateCloudsDeleteRequest object.

  Fields:
    delayHours: Optional. Time delay of the deletion specified in hours. The
      default value is `3`. Specifying a non-zero value for this field changes
      the value of `PrivateCloud.state` to `DELETED` and sets `expire_time` to
      the planned deletion time. Deletion can be cancelled before
      `expire_time` elapses using VmwareEngine.UndeletePrivateCloud.
      Specifying a value of `0` for this field instead begins the deletion
      process and ceases billing immediately. During the final deletion
      process, the value of `PrivateCloud.state` becomes `PURGING`.
    force: Optional. If set to true, cascade delete is enabled and all
      children of this private cloud resource are also deleted. When this flag
      is set to false, the private cloud will not be deleted if there are any
      children other than the management cluster. The management cluster is
      always deleted.
    name: Required. The resource name of the private cloud to delete. Resource
      names are schemeless URIs that follow the conventions in
      https://cloud.google.com/apis/design/resource_names. For example:
      `projects/my-project/locations/us-central1-a/privateClouds/my-cloud`
    requestId: Optional. The request ID must be a valid UUID with the
      exception that zero UUID is not supported
      (00000000-0000-0000-0000-000000000000).
  """
    delayHours = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    force = _messages.BooleanField(2)
    name = _messages.StringField(3, required=True)
    requestId = _messages.StringField(4)