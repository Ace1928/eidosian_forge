from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddLanugageArg(parser, required=False, help_text='preferred language of contact. Must be a valid ISO 639-1 language code.'):
    """Adds an arg for the contact's preferred language to the parser."""
    parser.add_argument('--language', help=help_text, required=required)