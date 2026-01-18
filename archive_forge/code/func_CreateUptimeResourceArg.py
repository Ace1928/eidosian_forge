from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def CreateUptimeResourceArg(verb):
    name = 'check_id'
    help_text = 'Name of the uptime check or synthetic monitor ' + verb
    return presentation_specs.ResourcePresentationSpec(name, GetUptimeCheckResourceSpec(), help_text, required=True)