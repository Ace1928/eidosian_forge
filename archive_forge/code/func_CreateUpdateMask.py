from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudkms import base as cloudkms_base
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.kms import certs
from googlecloudsdk.command_lib.kms import exceptions as kms_exceptions
from googlecloudsdk.command_lib.kms import flags
from googlecloudsdk.command_lib.kms import maps
from googlecloudsdk.command_lib.kms import resource_args
def CreateUpdateMask(self, args):
    update_mask = []
    if args.service_directory_service or args.endpoint_filter or args.hostname or args.server_certificates_files:
        update_mask.append('serviceResolvers')
    if args.key_management_mode:
        update_mask.append('keyManagementMode')
    if args.crypto_space_path:
        update_mask.append('cryptoSpacePath')
    return ','.join(update_mask)