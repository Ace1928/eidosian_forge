from __future__ import absolute_import
import sys
import os
from argparse import ArgumentParser, Action, SUPPRESS
from . import Options
class SetLenientAction(Action):

    def __call__(self, parser, namespace, values, option_string=None):
        namespace.error_on_unknown_names = False
        namespace.error_on_uninitialized = False