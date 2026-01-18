from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.recommender import flag_utils as api_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.recommender import flags
def AddArgsToParser(parser):
    """Add flags to parser.

  Args:
    parser: An argparse parser that you can use to add arguments that go on the
      command line after this command.
  """
    parser.add_argument('--project', metavar='PROJECT', required=True, help='Project number')
    parser.add_argument('--location', metavar='LOCATION', required=True, help='Location')
    parser.add_argument('--recommender', metavar='RECOMMENDER', required=True, help='Recommender for the recommender config')