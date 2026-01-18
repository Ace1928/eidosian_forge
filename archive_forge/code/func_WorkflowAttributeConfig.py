from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.core import properties
def WorkflowAttributeConfig():
    return concepts.ResourceParameterAttributeConfig(name='workflow', help_text='Workflow ID')