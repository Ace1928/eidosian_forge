import argparse
import re
from IPython.core.error import UsageError
from IPython.utils.decorators import undoc
from IPython.utils.process import arg_split
from IPython.utils.text import dedent
class MagicArgumentParser(argparse.ArgumentParser):
    """ An ArgumentParser tweaked for use by IPython magics.
    """

    def __init__(self, prog=None, usage=None, description=None, epilog=None, parents=None, formatter_class=MagicHelpFormatter, prefix_chars='-', argument_default=None, conflict_handler='error', add_help=False):
        if parents is None:
            parents = []
        super(MagicArgumentParser, self).__init__(prog=prog, usage=usage, description=description, epilog=epilog, parents=parents, formatter_class=formatter_class, prefix_chars=prefix_chars, argument_default=argument_default, conflict_handler=conflict_handler, add_help=add_help)

    def error(self, message):
        """ Raise a catchable error instead of exiting.
        """
        raise UsageError(message)

    def parse_argstring(self, argstring):
        """ Split a string into an argument list and parse that argument list.
        """
        argv = arg_split(argstring)
        return self.parse_args(argv)