from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def EntryTypeProjectAttributeConfig():
    return concepts.ResourceParameterAttributeConfig(name='entry-type-project', help_text='The project of the EntryType resource.')