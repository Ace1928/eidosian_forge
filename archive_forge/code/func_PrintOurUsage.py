from __future__ import print_function
import os
import re
import sys
def PrintOurUsage():
    """Print usage for the stub script."""
    print('Stub script %s (auto-generated). Options:' % sys.argv[0])
    print('--helpstub               Show help for stub script.')
    print('--debug_binary           Run python under debugger specified by --debugger.')
    print("--debugger=<debugger>    Debugger for --debug_binary. Default: 'gdb --args'.")
    print('--debug_script           Run wrapped script with python debugger module (pdb).')
    print('--show_command_and_exit  Print command which would be executed and exit.')
    print('These options must appear first in the command line, all others will be passed to the wrapped script.')