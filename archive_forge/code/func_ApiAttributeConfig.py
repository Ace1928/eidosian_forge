from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def ApiAttributeConfig(name='api', wildcard=False):
    fallthroughs = []
    if wildcard:
        fallthroughs.append(deps.Fallthrough(lambda: '-', 'Defaults to wildcard for all APIs'))
    return concepts.ResourceParameterAttributeConfig(name=name, fallthroughs=fallthroughs, help_text='API ID.')