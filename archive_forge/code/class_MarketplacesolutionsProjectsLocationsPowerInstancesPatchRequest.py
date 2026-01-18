from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MarketplacesolutionsProjectsLocationsPowerInstancesPatchRequest(_messages.Message):
    """A MarketplacesolutionsProjectsLocationsPowerInstancesPatchRequest
  object.

  Fields:
    name: Identifier. The resource name of this PowerInstance. Resource names
      are schemeless URIs that follow the conventions in
      https://cloud.google.com/apis/design/resource_names. Format: `projects/{
      project}/locations/{location}/powerInstances/{power_instance}`
    powerInstance: A PowerInstance resource to be passed as the request body.
    updateMask: Required. The list of fields to update. The currently
      supported fields are: - 'memory_gib' - 'virtual_cpu_cores'
  """
    name = _messages.StringField(1, required=True)
    powerInstance = _messages.MessageField('PowerInstance', 2)
    updateMask = _messages.StringField(3)