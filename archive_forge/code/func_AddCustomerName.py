from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.calliope import actions as calliope_actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddCustomerName(parser):
    """Adds customerName flag to the argparse.ArgumentParser."""
    parser.add_argument('--customer-name', help='    Customer name to put in the Letter of Authorization as the party\n    authorized to request an interconnect. This field is required for most\n    interconnects, however it is prohibited when creating a Cross-Cloud Interconnect.\n    ')