from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkeonpremProjectsLocationsBareMetalStandaloneClustersBareMetalStandaloneNodePoolsEnrollRequest(_messages.Message):
    """A GkeonpremProjectsLocationsBareMetalStandaloneClustersBareMetalStandalo
  neNodePoolsEnrollRequest object.

  Fields:
    enrollBareMetalStandaloneNodePoolRequest: A
      EnrollBareMetalStandaloneNodePoolRequest resource to be passed as the
      request body.
    parent: Required. The parent resource where this node pool will be
      created. projects/{project}/locations/{location}/bareMetalStandaloneClus
      ters/{cluster}
  """
    enrollBareMetalStandaloneNodePoolRequest = _messages.MessageField('EnrollBareMetalStandaloneNodePoolRequest', 1)
    parent = _messages.StringField(2, required=True)