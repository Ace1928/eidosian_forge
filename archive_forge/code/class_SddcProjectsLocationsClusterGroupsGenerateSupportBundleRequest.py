from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SddcProjectsLocationsClusterGroupsGenerateSupportBundleRequest(_messages.Message):
    """A SddcProjectsLocationsClusterGroupsGenerateSupportBundleRequest object.

  Fields:
    clusterGroup: Required. The resource name of the cluster group. For
      example, Format: `projects/{PROJECT-NUMBER}/locations/us-
      central1/clusterGroups/{MY_GROUP}`
    generateSupportBundleRequest: A GenerateSupportBundleRequest resource to
      be passed as the request body.
  """
    clusterGroup = _messages.StringField(1, required=True)
    generateSupportBundleRequest = _messages.MessageField('GenerateSupportBundleRequest', 2)