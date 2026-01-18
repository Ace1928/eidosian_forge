from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.deployment_manager import exceptions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.deployment_manager import importer
from googlecloudsdk.core import properties
def AddTemplateFlag(parser):
    """Add the template path argument.

  Args:
    parser: An argparse parser that you can use to add arguments that go
        on the command line after this command. Positional arguments are
        allowed.
  """
    parser.add_argument('--template', help='Path to a python or jinja file (local or via URL) that defines the composite type. If you want to provide a schema, that file must be in the same location: e.g. "--template=./foobar.jinja" means "./foobar.jinja.schema" should also exist. The file must end in either ".jinja" or ".py" to be interpreted correctly.', type=template_flag_arg_type, required=True)