from __future__ import absolute_import
import sys
import os
from argparse import ArgumentParser, Action, SUPPRESS
from . import Options
class SetAnnotateCoverageAction(Action):

    def __call__(self, parser, namespace, values, option_string=None):
        namespace.annotate = True
        namespace.annotate_coverage_xml = values