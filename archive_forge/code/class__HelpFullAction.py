from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import sys
from absl import flags
class _HelpFullAction(argparse.Action):
    """Action class for --helpfull flag."""

    def __init__(self, option_strings, dest, default, help):
        """Initializes _HelpFullAction.

    Args:
      option_strings: See argparse.Action.
      dest: Ignored. The flag is always defined with dest=argparse.SUPPRESS.
      default: Ignored.
      help: See argparse.Action.
    """
        del dest, default
        super(_HelpFullAction, self).__init__(option_strings=option_strings, dest=argparse.SUPPRESS, default=argparse.SUPPRESS, nargs=0, help=help)

    def __call__(self, parser, namespace, values, option_string=None):
        """See https://docs.python.org/3/library/argparse.html#action-classes."""
        parser.print_help()
        absl_flags = parser._inherited_absl_flags
        if absl_flags:
            modules = sorted(absl_flags.flags_by_module_dict())
            main_module = sys.argv[0]
            if main_module in modules:
                modules.remove(main_module)
            print(absl_flags._get_help_for_modules(modules, prefix='', include_special_flags=True))
        parser.exit()