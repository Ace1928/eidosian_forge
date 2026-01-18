from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddOperationNameFlag(parser, operation_type):
    """Adds a name flag to the given parser.

  Args:
    parser: The argparse parser.
    operation_type: The operate type displayed in help text, a str.
  """
    parser.add_argument('name', type=str, default=None, help='\n        The unique name of the Operation to {}, formatted as either the full\n        or relative resource path:\n\n          projects/my-app-id/operations/foo\n\n        or:\n\n          foo\n        '.format(operation_type))