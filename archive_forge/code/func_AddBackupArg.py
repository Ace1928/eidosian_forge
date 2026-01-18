from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def AddBackupArg(parser):
    concept_parsers.ConceptParser.ForResource('backup', GetBackupResourceSpec(), "\n      Name of the backup to create. Once the backup is created, this name can't\n      be changed. This must be 63 or fewer characters long and must be unique\n      within the project, location, and backup plan. The name may be provided\n      either as a relative name, e.g.\n      `projects/<project>/locations/<location>/backupPlans/<backupPlan>/backups/<backup>`\n      or as a single ID name (with the parent resources provided via options or\n      through properties), e.g.\n      `BACKUP --project=<project> --location=<location> --backup_plan=<backupPlan>`.\n      ", required=True).AddToParser(parser)