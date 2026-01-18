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
def AddContainerCommandField(parser):
    """Adds a --container-command flag to the given parser."""
    help_text = '  If set, overrides the default ENTRYPOINT specified by the image.\n\n  Example:\n\n    $ {command} --container-command=executable,parameter_1,parameter_2'
    parser.add_argument('--container-command', metavar='CONTAINER_COMMAND', type=arg_parsers.ArgList(), help=help_text)