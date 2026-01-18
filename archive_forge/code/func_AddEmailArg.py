from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddEmailArg(parser, required=False, help_text='email address of contact.'):
    """Adds an arg for the contact's email to the parser."""
    parser.add_argument('--email', help=help_text, required=required)