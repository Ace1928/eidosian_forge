from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import time
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import http_wrapper
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.container import constants as gke_constants
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources as cloud_resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import times
import six
from six.moves import range  # pylint: disable=redefined-builtin
import six.moves.http_client
from cmd argument to set a surge upgrade strategy.
def ParseClusterDNSOptions(self, options, is_update=False):
    """Parses the options for ClusterDNS."""
    if options.cluster_dns is None:
        if options.cluster_dns_scope:
            raise util.Error(PREREQUISITE_OPTION_ERROR_MSG.format(prerequisite='cluster-dns', opt='cluster-dns-scope'))
        if options.cluster_dns_domain:
            raise util.Error(PREREQUISITE_OPTION_ERROR_MSG.format(prerequisite='cluster-dns', opt='cluster-dns-domain'))
    if options.cluster_dns is None and options.cluster_dns_scope is None and (options.cluster_dns_domain is None) and (not is_update or options.disable_additive_vpc_scope is None) and (options.additive_vpc_scope_dns_domain is None):
        return
    dns_config = self.messages.DNSConfig()
    if options.cluster_dns is not None:
        provider_enum = self.messages.DNSConfig.ClusterDnsValueValuesEnum
        if options.cluster_dns.lower() == 'clouddns':
            dns_config.clusterDns = provider_enum.CLOUD_DNS
        elif options.cluster_dns.lower() == 'kubedns':
            dns_config.clusterDns = provider_enum.KUBE_DNS
        else:
            dns_config.clusterDns = provider_enum.PLATFORM_DEFAULT
    if options.cluster_dns_scope is not None:
        scope_enum = self.messages.DNSConfig.ClusterDnsScopeValueValuesEnum
        if options.cluster_dns_scope.lower() == 'cluster':
            dns_config.clusterDnsScope = scope_enum.CLUSTER_SCOPE
        else:
            dns_config.clusterDnsScope = scope_enum.VPC_SCOPE
    if options.cluster_dns_domain is not None:
        dns_config.clusterDnsDomain = options.cluster_dns_domain
    if options.additive_vpc_scope_dns_domain is not None:
        dns_config.additiveVpcScopeDnsDomain = options.additive_vpc_scope_dns_domain
    if is_update and options.disable_additive_vpc_scope:
        dns_config.additiveVpcScopeDnsDomain = ''
    return dns_config