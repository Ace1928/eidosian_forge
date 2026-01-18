from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import multitype
from googlecloudsdk.command_lib.secrets import completers as secrets_completers
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import resources
def AddCreateRotationGroup(parser):
    """Add flags for specifying rotation on secret creates."""
    group = parser.add_group(mutex=False, help='Rotation.')
    group.add_argument(_ArgOrFlag('next-rotation-time', False), help='Timestamp at which to send rotation notification.')
    group.add_argument(_ArgOrFlag('rotation-period', False), help='Duration of time (in seconds) between rotation notifications.')