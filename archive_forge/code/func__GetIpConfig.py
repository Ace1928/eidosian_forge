from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.database_migration import api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions as core_exceptions
def _GetIpConfig(self, args):
    ip_config = self.messages.SqlIpConfig(enableIpv4=args.enable_ip_v4, privateNetwork=args.private_network, requireSsl=args.require_ssl, authorizedNetworks=self._GetAuthorizedNetworks(args.authorized_networks))
    if self._api_version == 'v1':
        ip_config.allocatedIpRange = args.allocated_ip_range
    return ip_config