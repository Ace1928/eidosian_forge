from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
def OrganizationAttributeConfig():
    return concepts.ResourceParameterAttributeConfig(name='organization', help_text='The parent organization for the {resource}.')