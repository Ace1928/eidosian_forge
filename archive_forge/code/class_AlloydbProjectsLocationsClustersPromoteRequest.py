from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AlloydbProjectsLocationsClustersPromoteRequest(_messages.Message):
    """A AlloydbProjectsLocationsClustersPromoteRequest object.

  Fields:
    name: Required. The name of the resource. For the required format, see the
      comment on the Cluster.name field
    promoteClusterRequest: A PromoteClusterRequest resource to be passed as
      the request body.
  """
    name = _messages.StringField(1, required=True)
    promoteClusterRequest = _messages.MessageField('PromoteClusterRequest', 2)