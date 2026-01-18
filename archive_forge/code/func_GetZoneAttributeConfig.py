from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
def GetZoneAttributeConfig():
    return concepts.ResourceParameterAttributeConfig('zone', 'The zone of the {resource}.', fallthroughs=[deps.ArgFallthrough('region'), deps.ArgFallthrough('location'), deps.PropertyFallthrough(properties.VALUES.filestore.zone), deps.PropertyFallthrough(properties.VALUES.filestore.region), deps.PropertyFallthrough(properties.VALUES.filestore.location)])