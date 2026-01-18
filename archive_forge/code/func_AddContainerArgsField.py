from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def AddContainerArgsField(parser):
    """Adds a --container-args flag to the given parser."""
    help_text = '  Arguments passed to the entrypoint.\n\n  Example:\n\n    $ {command} --container-args=arg_1,arg_2'
    parser.add_argument('--container-args', metavar='CONTAINER_ARGS', type=arg_parsers.ArgList(), help=help_text)