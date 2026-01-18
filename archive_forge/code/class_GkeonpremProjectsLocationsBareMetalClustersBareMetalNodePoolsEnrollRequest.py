from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkeonpremProjectsLocationsBareMetalClustersBareMetalNodePoolsEnrollRequest(_messages.Message):
    """A
  GkeonpremProjectsLocationsBareMetalClustersBareMetalNodePoolsEnrollRequest
  object.

  Fields:
    enrollBareMetalNodePoolRequest: A EnrollBareMetalNodePoolRequest resource
      to be passed as the request body.
    parent: Required. The parent resource where this node pool will be
      created.
      projects/{project}/locations/{location}/bareMetalClusters/{cluster}
  """
    enrollBareMetalNodePoolRequest = _messages.MessageField('EnrollBareMetalNodePoolRequest', 1)
    parent = _messages.StringField(2, required=True)