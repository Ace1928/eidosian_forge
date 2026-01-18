from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import flags as shared_flags
from googlecloudsdk.command_lib.ai import region_util
from googlecloudsdk.command_lib.ai.custom_jobs import custom_jobs_util
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddCustomJobResourceArg(parser, verb, regions=constants.SUPPORTED_TRAINING_REGIONS):
    """Add a resource argument for a Vertex AI custom job.

  NOTE: Must be used only if it's the only resource arg in the command.

  Args:
    parser: the parser for the command.
    verb: str, the verb to describe the job resource, such as 'to update'.
    regions: list[str], the list of supported regions.
  """
    job_resource_spec = concepts.ResourceSpec(resource_collection=custom_jobs_util.CUSTOM_JOB_COLLECTION, resource_name='custom job', locationsId=shared_flags.RegionAttributeConfig(prompt_func=region_util.GetPromptForRegionFunc(regions)), projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG, disable_auto_completers=False)
    concept_parsers.ConceptParser.ForResource('custom_job', job_resource_spec, 'The custom job {}.'.format(verb), required=True).AddToParser(parser)