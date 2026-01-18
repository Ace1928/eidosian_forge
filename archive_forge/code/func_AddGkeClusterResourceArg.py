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
def AddGkeClusterResourceArg(parser):
    concept_parsers.ConceptParser.ForResource('--gke-cluster', _GetGkeClusterResourceSpec(), 'The GKE cluster to install the Dataproc cluster on.', required=True).AddToParser(parser)