from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def DeliveryPipelineAttributeConfig():
    """Creates the delivery pipeline resource attribute."""
    return concepts.ResourceParameterAttributeConfig(name='delivery-pipeline', fallthroughs=[deps.PropertyFallthrough(properties.FromString('deploy/delivery_pipeline'))], help_text='The delivery pipeline associated with the {resource}.  Alternatively, set the property [deploy/delivery-pipeline].')