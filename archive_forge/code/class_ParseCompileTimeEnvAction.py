from __future__ import absolute_import
import sys
import os
from argparse import ArgumentParser, Action, SUPPRESS
from . import Options
class ParseCompileTimeEnvAction(Action):

    def __call__(self, parser, namespace, values, option_string=None):
        old_env = dict(getattr(namespace, self.dest, {}))
        new_env = Options.parse_compile_time_env(values, current_settings=old_env)
        setattr(namespace, self.dest, new_env)