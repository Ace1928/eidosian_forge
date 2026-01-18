from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.deployment_manager import exceptions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.deployment_manager import importer
from googlecloudsdk.core import properties
def AddCompositeTypeNameFlag(parser):
    """Add the composite type name argument.

  Args:
    parser: An argparse parser that you can use to add arguments that go
        on the command line after this command. Positional arguments are
        allowed.
  """
    parser.add_argument('name', help='Composite type name.')