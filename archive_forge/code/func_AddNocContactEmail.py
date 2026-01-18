from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.calliope import actions as calliope_actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddNocContactEmail(parser):
    """Adds nocContactEmail flag to the argparse.ArgumentParser."""
    parser.add_argument('--noc-contact-email', help='      Email address to contact the customer NOC for operations and maintenance\n      notifications regarding this interconnect.\n      ')