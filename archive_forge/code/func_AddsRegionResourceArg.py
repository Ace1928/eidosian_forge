from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.cloudbuild import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddsRegionResourceArg(parser, is_required=True):
    """Add region resource argument to parser."""
    region_resource_spec = concepts.ResourceSpec('cloudbuild.projects.locations', resource_name='region', locationsId=resource_args.RegionAttributeConfig(), projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG)
    concept_parsers.ConceptParser.ForResource('--region', region_resource_spec, 'Region for Cloud Build.', required=is_required).AddToParser(parser)