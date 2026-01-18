from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.recommender import base
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.util.args import common_args
def AddDisplayNameToParser(parser, resource):
    """Adds display-name to parser.

  Args:
      parser: An argparse parser that you can use to add arguments that go on
        the command line after this command.
      resource: The resource to add to.
  """
    parser.add_argument('--display-name', help='Display name of the {}.'.format(resource))