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
def AddProjectsLocationsResourceArg(parser, api_version):
    """Add resrouce arg for projects/{}/locations/{}."""
    spec = concepts.ResourceSpec('dataproc.projects.locations', api_version=api_version, resource_name='region', disable_auto_completers=True, projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG, locationsId=_RegionAttributeConfig())
    concept_parsers.ConceptParser.ForResource('--region', spec, properties.VALUES.dataproc.region.help_text, required=True).AddToParser(parser)