from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import json
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
import six
def AddArgs(parser):
    """Remaining args to the program."""
    parser.add_argument('args', metavar='JOB_ARG', nargs=argparse.REMAINDER, default=[], help='Arguments to pass to the driver.')