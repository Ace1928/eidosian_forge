from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.run import connection_context
from googlecloudsdk.command_lib.run import exceptions
from googlecloudsdk.command_lib.run import flags
from googlecloudsdk.command_lib.run import platforms
from googlecloudsdk.command_lib.run import resource_args
from googlecloudsdk.command_lib.run import serverless_operations
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
@base.ReleaseTracks(base.ReleaseTrack.BETA)
class BetaDescribe(Describe):
    """Describe domain mappings."""
    detailed_help = {'DESCRIPTION': '{description}', 'EXAMPLES': '          To describe a Cloud Run domain mapping, run:\n\n              $ {command} --domain=www.example.com\n          '}

    @staticmethod
    def Args(parser):
        Describe.CommonArgs(parser)