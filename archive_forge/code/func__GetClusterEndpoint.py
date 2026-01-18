from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import os
import re
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.container import kubeconfig as kconfig
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.api_lib.services import exceptions
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import platforms
import six
def _GetClusterEndpoint(cluster, use_internal_ip, cross_connect_subnetwork, use_private_fqdn, use_dns_endpoint):
    """Get the cluster endpoint suitable for writing to kubeconfig."""
    if use_dns_endpoint:
        return _GetDNSEndpoint(cluster)
    if use_internal_ip or cross_connect_subnetwork or use_private_fqdn:
        if not cluster.privateClusterConfig:
            raise NonPrivateClusterError(cluster)
        if not cluster.privateClusterConfig.privateEndpoint:
            raise MissingPrivateEndpointError(cluster)
        if cross_connect_subnetwork is not None:
            return _GetCrossConnectSubnetworkEndpoint(cluster, cross_connect_subnetwork)
        if use_private_fqdn:
            return _GetFqdnPrivateEndpoint(cluster)
        return cluster.privateClusterConfig.privateEndpoint
    if not cluster.endpoint:
        raise MissingEndpointError(cluster)
    return cluster.endpoint