from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import cdn_flags_utils as cdn_flags
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import reference_utils
from googlecloudsdk.command_lib.compute import signed_url_flags
from googlecloudsdk.command_lib.compute.backend_services import backend_services_utils
from googlecloudsdk.command_lib.compute.backend_services import flags
from googlecloudsdk.core import log
def _ResolveProtocol(messages, args, default='HTTP'):
    valid_options = messages.BackendService.ProtocolValueValuesEnum.names()
    if args.protocol and args.protocol not in valid_options:
        raise ValueError('{} is not a supported option. See the help text of --protocol for supported options.'.format(args.protocol))
    return messages.BackendService.ProtocolValueValuesEnum(args.protocol or default)