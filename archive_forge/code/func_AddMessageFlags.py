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
def AddMessageFlags(parser, resource, flag=None):
    """Adds flags for specifying a message as a string/file to the parser."""
    message_group = parser.add_group(mutex=True)
    message_group.add_argument('--{}'.format(flag or resource), help='The {} as a string. In either JSON or YAML format.'.format(resource))
    message_group.add_argument('--{}-from-file'.format(flag or resource), type=arg_parsers.FileContents(), help='The path to a JSON or YAML file containing the {}.'.format(resource))