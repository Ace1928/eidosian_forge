import glob
import os
import struct
from .lazy_import import lazy_import
import ctypes
from breezy import cmdline
from breezy.i18n import gettext
def _command_line_to_argv(command_line, argv, single_quotes_allowed=False):
    """Convert a Unicode command line into a list of argv arguments.

    It performs wildcard expansion to make wildcards act closer to how they
    work in posix shells, versus how they work by default on Windows. Quoted
    arguments are left untouched.

    :param command_line: The unicode string to split into an arg list.
    :param single_quotes_allowed: Whether single quotes are accepted as quoting
                                  characters like double quotes. False by
                                  default.
    :return: A list of unicode strings.
    """
    s = cmdline.Splitter(command_line, single_quotes_allowed=single_quotes_allowed)
    arguments = list(s)
    if len(arguments) < len(argv):
        raise AssertionError("Split command line can't be shorter than argv")
    arguments = arguments[len(arguments) - len(argv):]
    args = []
    for is_quoted, arg in arguments:
        if is_quoted or not glob.has_magic(arg):
            args.append(arg)
        else:
            args.extend(glob_one(arg))
    return args