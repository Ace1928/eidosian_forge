from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
def TPUAttributeConfig():
    return concepts.ResourceParameterAttributeConfig(name='TPU', help_text='The TPU Name for the {resource}.')