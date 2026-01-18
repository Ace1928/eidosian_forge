from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.backupdr import util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddBackupPlanResourceArg(parser, help_text):
    """Adds an argument for backup plan to parser."""
    name = 'backup_plan'
    backup_plan_spec = concepts.ResourceSpec('backupdr.projects.locations.backupPlans', resource_name='Backup Plan', locationsId=LocationAttributeConfig(), projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG, disable_auto_completers=False)
    concept_parsers.ConceptParser.ForResource(name, backup_plan_spec, help_text, required=True).AddToParser(parser)