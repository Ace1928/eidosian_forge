from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import collections
from gslib.exception import CommandException
class HelpProvider(object):
    """Interface for providing help."""
    HelpSpec = collections.namedtuple('HelpSpec', ['help_name', 'help_name_aliases', 'help_type', 'help_one_line_summary', 'help_text', 'subcommand_help_text'])
    help_spec = None