from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkeonpremProjectsLocationsBareMetalStandaloneClustersBareMetalStandaloneNodePoolsGetRequest(_messages.Message):
    """A GkeonpremProjectsLocationsBareMetalStandaloneClustersBareMetalStandalo
  neNodePoolsGetRequest object.

  Fields:
    name: Required. The name of the bare metal standalone node pool to
      retrieve. projects/{project}/locations/{location}/bareMetalStandaloneClu
      sters/{cluster}/bareMetalStandaloneNodePools/{nodepool}
  """
    name = _messages.StringField(1, required=True)