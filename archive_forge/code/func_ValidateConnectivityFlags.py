from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def ValidateConnectivityFlags(args):
    """Validate the arguments for connectivity, ensure the correct set of flags are passed."""
    if args.enable_private_service_connect and args.network:
        raise exceptions.ConflictingArgumentsException('--enable-private-service-connect', '--network')
    if args.enable_private_service_connect and args.allocated_ip_range_name:
        raise exceptions.ConflictingArgumentsException('--enable-private-service-connect', '--allocated-ip-range-name')