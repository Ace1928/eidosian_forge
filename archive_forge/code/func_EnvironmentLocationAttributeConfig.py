from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def EnvironmentLocationAttributeConfig(fallthroughs_enabled=True):
    fallthroughs = [deps.PropertyFallthrough(properties.VALUES.composer.location)] if fallthroughs_enabled else []
    return concepts.ResourceParameterAttributeConfig(name='location', help_text='Region where Composer environment runs or in which to create the environment.', fallthroughs=fallthroughs)