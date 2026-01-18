import errno as _errno
import sys as _sys
from . import flags
def _define_help_flags():
    global _define_help_flags_called
    if not _define_help_flags_called:
        flags.DEFINE_flag(_HelpFlag())
        flags.DEFINE_flag(_HelpfullFlag())
        flags.DEFINE_flag(_HelpshortFlag())
        _define_help_flags_called = True