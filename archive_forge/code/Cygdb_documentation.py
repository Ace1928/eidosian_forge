import os
import sys
import glob
import tempfile
import textwrap
import subprocess
import optparse
import logging

    Start the Cython debugger. This tells gdb to import the Cython and Python
    extensions (libcython.py and libpython.py) and it enables gdb's pending
    breakpoints.

    path_to_debug_info is the path to the Cython build directory
    gdb_argv is the list of options to gdb
    no_import tells cygdb whether it should import debug information
    