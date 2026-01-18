from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.netapp import constants
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
def AddResourceDescriptionArg(parser, resource_name):
    """Add Description arg to arg_parser for a resource called resource_name."""
    parser.add_argument('--description', required=False, help='A description of the Cloud NetApp {}'.format(resource_name))