from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.base import ReleaseTrack
from googlecloudsdk.command_lib.assured import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddWorkloadResourceArgToParser(parser, verb):
    concept_parsers.ConceptParser.ForResource('workload', resource_args.GetWorkloadResourceSpec(), 'The Assured Workloads environment resource to {}.'.format(verb), required=True).AddToParser(parser)