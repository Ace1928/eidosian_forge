from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import ipaddress
import re
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute.instance_templates import service_proxy_aux_data
def AddPostKeyRevocationActionTypeArgs(parser):
    """Helper to add --post-key-revocation-action-type flag."""
    help_text = 'Specifies the behavior of the instance when the KMS key of one of its attached disks is revoked. The default is noop.'
    choices_text = {'noop': 'No operation is performed.', 'shutdown': 'The instance is shut down when the KMS key of one of its attached disks is revoked.'}
    parser.add_argument('--post-key-revocation-action-type', choices=choices_text, metavar='POLICY', required=False, help=help_text)