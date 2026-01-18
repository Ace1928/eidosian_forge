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
def AddContainerEnvField(parser):
    """Adds a --container-env flag to the given parser."""
    help_text = '  Environment variables passed to the container.\n\n  Example:\n\n    $ {command} --container-env=key1=value1,key2=value2'
    parser.add_argument('--container-env', metavar='CONTAINER_ENV', type=arg_parsers.ArgDict(key_type=str, value_type=str), help=help_text)