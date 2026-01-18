from __future__ import absolute_import
import sys
import os
from argparse import ArgumentParser, Action, SUPPRESS
from . import Options
class ParseOptionsAction(Action):

    def __call__(self, parser, namespace, values, option_string=None):
        options = dict(getattr(namespace, self.dest, {}))
        for opt in values.split(','):
            if '=' in opt:
                n, v = opt.split('=', 1)
                v = v.lower() not in ('false', 'f', '0', 'no')
            else:
                n, v = (opt, True)
            options[n] = v
        setattr(namespace, self.dest, options)