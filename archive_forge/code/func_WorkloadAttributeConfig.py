from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
def WorkloadAttributeConfig():
    return concepts.ResourceParameterAttributeConfig(name='workload', help_text='The workload for the {resource}.')