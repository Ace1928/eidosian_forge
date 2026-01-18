from __future__ import absolute_import
import sys
import os
from argparse import ArgumentParser, Action, SUPPRESS
from . import Options
class SetGDBDebugOutputAction(Action):

    def __call__(self, parser, namespace, values, option_string=None):
        namespace.gdb_debug = True
        namespace.output_dir = values