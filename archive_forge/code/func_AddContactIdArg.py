from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddContactIdArg(parser, help_text='id of contact'):
    """Adds an arg for the contact id to the parser."""
    parser.add_argument('CONTACT_ID', type=str, help=help_text)