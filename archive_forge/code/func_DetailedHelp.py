from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.operations import flags
from googlecloudsdk.core import resources
def DetailedHelp():
    """Construct help text based on the command release track."""
    detailed_help = {'brief': 'Describe a Compute Engine operation', 'DESCRIPTION': '\n        *{command}* displays all data associated with a Compute Engine\n        operation in a project.\n        ', 'EXAMPLES': '\n        To get details about a global operation (e.g. operation-111-222-333-444), run:\n\n          $ {command} operation-111-222-333-444 --global\n\n        To get details about a regional operation, run:\n\n          $ {command} operation-111-222-333-444 --region=us-central1\n\n        To get details about a zonal operation, run:\n\n          $ {command} operation-111-222-333-444 --zone=us-central1-a\n        '}
    return detailed_help