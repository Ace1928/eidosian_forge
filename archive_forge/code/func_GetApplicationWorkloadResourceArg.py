from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.apphub import utils as apphub_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def GetApplicationWorkloadResourceArg(arg_name='workload', help_text=None, positional=True, required=True):
    """Constructs and returns the application workload Resource Argument."""
    help_text = help_text or 'The Workload ID.'
    return concept_parsers.ConceptParser.ForResource('{}{}'.format('' if positional else '--', arg_name), GetApplicationWorkloadResourceSpec(), help_text, required=required)