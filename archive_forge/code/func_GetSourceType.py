from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.vmware import util
from googlecloudsdk.command_lib.util.apis import arg_utils
def GetSourceType(self, source_type):
    source_type_enum = arg_utils.ChoiceEnumMapper(arg_name='source_type', message_enum=self.messages.LoggingServer.SourceTypeValueValuesEnum, include_filter=lambda x: 'SOURCE_TYPE_UNSPECIFIED' not in x).GetEnumForChoice(arg_utils.EnumNameToChoice(source_type))
    return source_type_enum