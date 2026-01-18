from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.spanner import database_sessions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.spanner import resource_args
from googlecloudsdk.command_lib.spanner import sql
from googlecloudsdk.command_lib.spanner.sql import QueryHasDml
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def GetRequestPriorityMapper(messages):
    return arg_utils.ChoiceEnumMapper('--priority', messages.RequestOptions.PriorityValueValuesEnum, custom_mappings={'PRIORITY_LOW': 'low', 'PRIORITY_MEDIUM': 'medium', 'PRIORITY_HIGH': 'high', 'PRIORITY_UNSPECIFIED': 'unspecified'}, help_str='The priority for the execute SQL request.')