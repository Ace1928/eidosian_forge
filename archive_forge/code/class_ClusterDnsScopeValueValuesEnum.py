from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClusterDnsScopeValueValuesEnum(_messages.Enum):
    """cluster_dns_scope indicates the scope of access to cluster DNS
    records.

    Values:
      DNS_SCOPE_UNSPECIFIED: Default value, will be inferred as cluster scope.
      CLUSTER_SCOPE: DNS records are accessible from within the cluster.
      VPC_SCOPE: DNS records are accessible from within the VPC.
    """
    DNS_SCOPE_UNSPECIFIED = 0
    CLUSTER_SCOPE = 1
    VPC_SCOPE = 2