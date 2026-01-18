from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def AddRestoreArg(parser):
    concept_parsers.ConceptParser.ForResource('restore', GetRestoreResourceSpec(), "\n      Name of the restore to create. Once the restore is created, this name\n      can't be changed. This must be 63 or fewer characters long and must be\n      unique within the project and location. The name may be provided either as\n      a relative name, e.g.\n      `projects/<project>/locations/<location>/restorePlans/<restorePlan>/restores/<restore>`\n      or as a single ID name (with the parent resources provided via options or\n      through properties), e.g.\n      `<restore> --project=<project> --location=<location> --restore-plan=<restorePlan>`.\n      ", required=True).AddToParser(parser)