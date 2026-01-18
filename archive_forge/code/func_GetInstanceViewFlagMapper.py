from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def GetInstanceViewFlagMapper(alloydb_messages):
    return arg_utils.ChoiceEnumMapper('--view', alloydb_messages.AlloydbProjectsLocationsClustersInstancesGetRequest.ViewValueValuesEnum, required=False, custom_mappings={'INSTANCE_VIEW_BASIC': 'basic', 'INSTANCE_VIEW_FULL': 'full'}, help_str='View of the instance to return.')