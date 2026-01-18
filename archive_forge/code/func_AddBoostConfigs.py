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
def AddBoostConfigs(parser):
    """Adds a --boost-config flag to the given parser."""
    help_text = '  Boost Configuration(s) that workstations running with this configuration can\n  boost up to. This includes id (required), machine-type, accelerator-type, and\n  accelerator-count.\n\n  Example:\n\n    $ {command} --boost-config=id=boost1,machine-type=n1-standard-4,accelerator-type=nvidia-tesla-t4,accelerator-count=1'
    parser.add_argument('--boost-config', metavar='BOOST_CONFIG', type=arg_parsers.ArgObject(spec={'id': str, 'machine-type': str, 'accelerator-type': str, 'accelerator-count': int}, required_keys=['id']), action=arg_parsers.FlattenAction(), help=help_text)