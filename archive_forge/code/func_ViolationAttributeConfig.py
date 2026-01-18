from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
def ViolationAttributeConfig():
    return concepts.ResourceParameterAttributeConfig(name='violation', help_text='The violation for the {resource}.')