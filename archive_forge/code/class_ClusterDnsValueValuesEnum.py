from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClusterDnsValueValuesEnum(_messages.Enum):
    """cluster_dns indicates which in-cluster DNS provider should be used.

    Values:
      PROVIDER_UNSPECIFIED: Default value
      PLATFORM_DEFAULT: Use GKE default DNS provider(kube-dns) for DNS
        resolution.
      CLOUD_DNS: Use CloudDNS for DNS resolution.
      KUBE_DNS: Use KubeDNS for DNS resolution.
    """
    PROVIDER_UNSPECIFIED = 0
    PLATFORM_DEFAULT = 1
    CLOUD_DNS = 2
    KUBE_DNS = 3