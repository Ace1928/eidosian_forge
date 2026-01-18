from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.monitoring import completers
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.args import repeated
from googlecloudsdk.core.util import times
def AddFileMessageFlag(parser, resource, flag=None):
    """Adds flags for specifying a message as a file to the parser."""
    parser.add_argument('--{}-from-file'.format(flag or resource), type=arg_parsers.FileContents(), help='The path to a JSON or YAML file containing the {}.'.format(resource))