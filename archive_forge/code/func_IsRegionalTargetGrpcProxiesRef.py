from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import scope as compute_scope
def IsRegionalTargetGrpcProxiesRef(target_grpc_proxy_ref):
    """Returns True if the target gRPC proxy reference is regional."""
    return target_grpc_proxy_ref.Collection() == 'compute.regionTargetGrpcProxies'