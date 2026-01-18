from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EdgecontainerProjectsLocationsClustersGenerateAccessTokenRequest(_messages.Message):
    """A EdgecontainerProjectsLocationsClustersGenerateAccessTokenRequest
  object.

  Fields:
    cluster: Required. The resource name of the cluster.
  """
    cluster = _messages.StringField(1, required=True)