from __future__ import absolute_import
import sys
import os
from argparse import ArgumentParser, Action, SUPPRESS
from . import Options
class ParseDirectivesAction(Action):

    def __call__(self, parser, namespace, values, option_string=None):
        old_directives = dict(getattr(namespace, self.dest, Options.get_directive_defaults()))
        directives = Options.parse_directive_list(values, relaxed_bool=True, current_settings=old_directives)
        setattr(namespace, self.dest, directives)