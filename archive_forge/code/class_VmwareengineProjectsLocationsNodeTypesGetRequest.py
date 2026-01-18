from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareengineProjectsLocationsNodeTypesGetRequest(_messages.Message):
    """A VmwareengineProjectsLocationsNodeTypesGetRequest object.

  Fields:
    name: Required. The resource name of the node type to retrieve. Resource
      names are schemeless URIs that follow the conventions in
      https://cloud.google.com/apis/design/resource_names. For example:
      `projects/my-proj/locations/us-central1-a/nodeTypes/standard-72`
  """
    name = _messages.StringField(1, required=True)