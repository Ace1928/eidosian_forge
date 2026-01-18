from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareengineProjectsLocationsPrivateCloudsClustersCreateRequest(_messages.Message):
    """A VmwareengineProjectsLocationsPrivateCloudsClustersCreateRequest
  object.

  Fields:
    cluster: A Cluster resource to be passed as the request body.
    clusterId: Required. The user-provided identifier of the new `Cluster`.
      This identifier must be unique among clusters within the parent and
      becomes the final token in the name URI. The identifier must meet the
      following requirements: * Only contains 1-63 alphanumeric characters and
      hyphens * Begins with an alphabetical character * Ends with a non-hyphen
      character * Not formatted as a UUID * Complies with [RFC
      1034](https://datatracker.ietf.org/doc/html/rfc1034) (section 3.5)
    parent: Required. The resource name of the private cloud to create a new
      cluster in. Resource names are schemeless URIs that follow the
      conventions in https://cloud.google.com/apis/design/resource_names. For
      example: `projects/my-project/locations/us-central1-a/privateClouds/my-
      cloud`
    requestId: Optional. The request ID must be a valid UUID with the
      exception that zero UUID is not supported
      (00000000-0000-0000-0000-000000000000).
    validateOnly: Optional. True if you want the request to be validated and
      not executed; false otherwise.
  """
    cluster = _messages.MessageField('Cluster', 1)
    clusterId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    requestId = _messages.StringField(4)
    validateOnly = _messages.BooleanField(5)