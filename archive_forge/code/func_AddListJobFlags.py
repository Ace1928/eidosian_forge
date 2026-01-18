from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddListJobFlags(parser):
    """Add job list Flags."""
    GetLocationResourceArg(required=True).AddToParser(parser)
    concept_parsers.ConceptParser([presentation_specs.ResourcePresentationSpec('--experiment', GetExperimentResourceSpec(), 'The experiment resource.', prefixes=True, required=False), presentation_specs.ResourcePresentationSpec('--fault', GetFaultResourceSpec(), 'The fault resource.', prefixes=True, required=False)]).AddToParser(parser)
    parser.add_argument('--target-name', type=str, help='target name.', required=False)
    parser.add_argument('--days', type=int, help='Days.', required=False)