from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AlloydbProjectsLocationsClustersRestoreRequest(_messages.Message):
    """A AlloydbProjectsLocationsClustersRestoreRequest object.

  Fields:
    parent: Required. The name of the parent resource. For the required
      format, see the comment on the Cluster.name field.
    restoreClusterRequest: A RestoreClusterRequest resource to be passed as
      the request body.
  """
    parent = _messages.StringField(1, required=True)
    restoreClusterRequest = _messages.MessageField('RestoreClusterRequest', 2)