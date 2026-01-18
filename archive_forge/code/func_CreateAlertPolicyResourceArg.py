from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def CreateAlertPolicyResourceArg(verb, positional=True):
    if positional:
        name = 'alert_policy'
    else:
        name = '--policy'
    help_text = 'Name of the Alert Policy ' + verb
    return presentation_specs.ResourcePresentationSpec(name, GetAlertPolicyResourceSpec(), help_text, required=True)