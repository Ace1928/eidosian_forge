from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.network_services import completers as network_services_completers
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.apis import arg_utils
def BackendServiceArgumentForUrlMapPathMatcher(required=True):
    return compute_flags.ResourceArgument(resource_name='backend service', name='--default-service', required=required, completer=BackendServicesCompleter, global_collection='compute.backendServices', short_help='A backend service that will be used for requests that the path matcher cannot match.')