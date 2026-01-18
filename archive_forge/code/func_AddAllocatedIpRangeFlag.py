from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddAllocatedIpRangeFlag(parser):
    """Adds a --allocated-ip-range flag to the given parser."""
    help_text = '    The name of the allocated IP range for the private IP Cloud SQL instance.\n    This name refers to an already allocated IP range.\n    If set, the instance IP will be created in the allocated range.\n  '
    parser.add_argument('--allocated-ip-range', help=help_text)