from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def GetFaultResourceArg(arg_name='fault', help_text=None, positional=True, required=True):
    """Constructs and returns the Fault Resource Argument."""
    help_text = help_text or 'Name for the Fault'
    return concept_parsers.ConceptParser.ForResource('{}{}'.format('' if positional else '--', arg_name), GetFaultResourceSpec(), help_text, required=required)