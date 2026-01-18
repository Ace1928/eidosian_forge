import argparse
import enum
import logging
import os
import shlex
import subprocess
import sys
from typing import Optional
import warnings
class CExtensionOptions(object):
    """Options to compile C extensions."""

    def __init__(self):
        self.extra_link_args = []
        self.extra_compile_args = ['-std=c99']
        self.include_dirs = []
        self.libraries = []
        self.library_dirs = []

    def add_include(self, args, unknown):
        """Add include directories.

        :param args: args as returned by get_r_flags().
        :param unknown: unknown arguments a returned by get_r_flags()."""
        if args.I is None:
            warnings.warn('No include specified')
        else:
            self.include_dirs.extend(args.I)
        self.extra_compile_args.extend(unknown)

    def add_lib(self, args, unknown, ignore=('R',)):
        """Add libraries.

        :param args: args as returned by get_r_flags().
        :param unknown: unknown arguments a returned by get_r_flags()."""
        if args.L is None:
            if args.l is None:
                warnings.warn('No libraries as -l arguments to the compiler.')
            else:
                self.libraries.extend([x for x in args.l if x not in ignore])
        else:
            self.library_dirs.extend(args.L)
            self.libraries.extend(args.l)
        self.extra_link_args.extend(unknown)