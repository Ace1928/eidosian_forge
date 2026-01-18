from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AnthosCluster(_messages.Message):
    """Information specifying an Anthos Cluster.

  Fields:
    membership: Membership of the GKE Hub-registered cluster to which to apply
      the Skaffold configuration. Format is
      `projects/{project}/locations/{location}/memberships/{membership_name}`.
  """
    membership = _messages.StringField(1)