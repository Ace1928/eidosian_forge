from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.notebooks import environments as env_util
from googlecloudsdk.api_lib.notebooks import util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def CreateInstanceMigrateRequest(args, messages):
    """"Create and return Migrate request."""
    instance = GetInstanceResource(args).RelativeName()

    def GetPostStartupScriptOption():
        type_enum = None
        if args.IsSpecified('post_startup_script_option'):
            request_message = messages.MigrateInstanceRequest
            type_enum = arg_utils.ChoiceEnumMapper(arg_name='post-startup-script-option', message_enum=request_message.PostStartupScriptOptionValueValuesEnum, include_filter=lambda x: 'UNSPECIFIED' not in x).GetEnumForChoice(arg_utils.EnumNameToChoice(args.post_startup_script_option))
        return type_enum
    return messages.NotebooksProjectsLocationsInstancesMigrateRequest(name=instance, migrateInstanceRequest=messages.MigrateInstanceRequest(postStartupScriptOption=GetPostStartupScriptOption()))