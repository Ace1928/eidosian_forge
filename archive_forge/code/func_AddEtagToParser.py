from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.recommender import base
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.util.args import common_args
def AddEtagToParser(parser, resource):
    """Adds etag to parser.

  Args:
      parser: An argparse parser that you can use to add arguments that go on
        the command line after this command.
      resource: The resource to add to.
  """
    parser.add_argument('--etag', required=True, help='Etag of the {}.'.format(resource))