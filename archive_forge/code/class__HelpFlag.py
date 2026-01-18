import errno as _errno
import sys as _sys
from . import flags
class _HelpFlag(flags.BooleanFlag):
    """Special boolean flag that displays usage and raises SystemExit."""
    NAME = 'help'
    SHORT_NAME = 'h'

    def __init__(self):
        super().__init__(self.NAME, False, 'show this help', short_name=self.SHORT_NAME)

    def parse(self, arg):
        if arg:
            _usage(shorthelp=True)
            print()
            print('Try --helpfull to get a list of all flags.')
            _sys.exit(1)