from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddCreateJobFlags(parser):
    """Add job Create Flags."""
    concept_parsers.ConceptParser([presentation_specs.ResourcePresentationSpec('JOB', GetJobResourceSpec(), 'The Job resource.', flag_name_overrides={'location': '--location'}, prefixes=True, required=True), presentation_specs.ResourcePresentationSpec('--experiment', GetExperimentResourceSpec(), 'The experiment resource.', flag_name_overrides={'location': '', 'project': ''}, prefixes=True, required=True)]).AddToParser(parser)
    parser.add_argument('--fault-targets', type=arg_parsers.ArgList(), metavar='LIST', help='targets for the faults in experiment. Provide them in sequence', required=True)
    parser.add_argument('--dry-run', action='store_true', default=False, help='Dry run mode.')