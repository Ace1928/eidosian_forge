from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddOrganization(parser, required=True):
    parser.add_argument('--organization', required=required, help='Organization which the organization firewall policy belongs to. Must be set if FIREWALL_POLICY is short name.')