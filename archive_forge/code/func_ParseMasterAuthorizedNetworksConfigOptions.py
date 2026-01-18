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
def ParseMasterAuthorizedNetworksConfigOptions(self, args, release_track):
    if args.enable_master_authorized_networks:
        self.enable_master_authorized_networks = args.enable_master_authorized_networks
    elif args.master_authorized_networks:
        raise command_util.InvalidUserInputError('Cannot specify --master-authorized-networks without ' + '--enable-master-authorized-networks.')
    command_util.ValidateMasterAuthorizedNetworks(args.master_authorized_networks)
    self.master_authorized_networks = args.master_authorized_networks