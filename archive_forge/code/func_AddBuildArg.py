from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.util import completers
def AddBuildArg(parser, intro=None):
    """Adds a 'build' arg to the given parser.

  Args:
    parser: The argparse parser to add the arg to.
    intro: Introductory sentence.
  """
    if intro:
        help_text = intro + ' '
    else:
        help_text = ''
    help_text += 'The ID of the build is printed at the end of the build submission process, or in the ID column when listing builds.'
    parser.add_argument('build', completer=BuildsCompleter, help=help_text)