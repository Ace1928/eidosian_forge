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
def AddReplicaZones(parser):
    """Adds a --replica-zones flag to the given parser."""
    help_text = "  Specifies the zones the VM and disk resources will be\n  replicated within the region. If set, exactly two zones within the\n  workstation cluster's region must be specified.\n\n  Example:\n\n    $ {command} --replica-zones=us-central1-a,us-central1-f\n  "
    parser.add_argument('--replica-zones', metavar='REPLICA_ZONES', type=arg_parsers.ArgList(), help=help_text)