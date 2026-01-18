from distutils import cmd
import distutils.errors
import logging
import os
import sys
import warnings
def _run_testr(self, *args):
    logger.debug('_run_testr called with args = %r', args)
    return commands.run_argv([sys.argv[0]] + list(args), sys.stdin, sys.stdout, sys.stderr)