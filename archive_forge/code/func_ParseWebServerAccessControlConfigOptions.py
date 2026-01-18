from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.composer import environments_util as environments_api_util
from googlecloudsdk.api_lib.composer import operations_util as operations_api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import flags
from googlecloudsdk.command_lib.composer import image_versions_util
from googlecloudsdk.command_lib.composer import parsers
from googlecloudsdk.command_lib.composer import resource_args
from googlecloudsdk.command_lib.composer import util as command_util
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
import six
def ParseWebServerAccessControlConfigOptions(self, args):
    if args.enable_private_environment and (not args.web_server_allow_ip) and (not args.web_server_allow_all) and (not args.web_server_deny_all):
        raise command_util.InvalidUserInputError('Cannot specify --enable-private-environment without one of: ' + '--web-server-allow-ip, --web-server-allow-all ' + 'or --web-server-deny-all')
    self.web_server_access_control = environments_api_util.BuildWebServerAllowedIps(args.web_server_allow_ip, args.web_server_allow_all or not args.web_server_allow_ip, args.web_server_deny_all)
    flags.ValidateIpRanges([acl['ip_range'] for acl in self.web_server_access_control])