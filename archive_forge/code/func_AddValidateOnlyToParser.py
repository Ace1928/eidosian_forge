from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.recommender import base
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.util.args import common_args
def AddValidateOnlyToParser(parser):
    """Adds validate-only to parser.

  Args:
      parser: An argparse parser that you can use to add arguments that go on
        the command line after this command.
  """
    parser.add_argument('--validate-only', action='store_true', default=False, help='If true, validate the request and preview the change, but do not actually update it.')