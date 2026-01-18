from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.accesscontextmanager import util
from googlecloudsdk.api_lib.accesscontextmanager import zones as zones_api
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.accesscontextmanager import perimeters
from googlecloudsdk.command_lib.accesscontextmanager import policies
from googlecloudsdk.command_lib.util.args import repeated
def _IsFieldSpecified(field_name, args):
    list_command_prefixes = ['remove_', 'add_', 'clear_']
    list_args = [command + field_name for command in list_command_prefixes]
    return any((args.IsSpecified(arg) for arg in list_args))