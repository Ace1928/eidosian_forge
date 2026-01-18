from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import scope as compute_scope
def IsRegionalTargetSslProxiesRef(target_ssl_proxy_ref):
    """Returns True if the Target SSL Proxy reference is regional."""
    return target_ssl_proxy_ref.Collection() == 'compute.regionTargetSslProxies'