import doctest
import errno
import glob
import logging
import os
import shlex
import sys
import textwrap
from .. import osutils, tests, trace
from ..tests import ui_testing
def _scan_redirection_options(args):
    """Recognize and process input and output redirections.

    :param args: The command line arguments

    :return: A tuple containing:
        - The file name redirected from or None
        - The file name redirected to or None
        - The mode to open the output file or None
        - The reamining arguments
    """

    def redirected_file_name(direction, name, args):
        if name == '':
            try:
                name = args.pop(0)
            except IndexError:
                name = ''
        return name
    remaining = []
    in_name = None
    out_name, out_mode = (None, None)
    while args:
        arg = args.pop(0)
        if arg.startswith('<'):
            in_name = redirected_file_name('<', arg[1:], args)
        elif arg.startswith('>>'):
            out_name = redirected_file_name('>>', arg[2:], args)
            out_mode = 'a+'
        elif arg.startswith('>'):
            out_name = redirected_file_name('>', arg[1:], args)
            out_mode = 'w+'
        else:
            remaining.append(arg)
    return (in_name, out_name, out_mode, remaining)