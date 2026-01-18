from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddDeliveryPipeline(parser, required=True):
    """Adds delivery pipeline flag."""
    parser.add_argument('--delivery-pipeline', help='The name of the Cloud Deploy delivery pipeline', required=required)