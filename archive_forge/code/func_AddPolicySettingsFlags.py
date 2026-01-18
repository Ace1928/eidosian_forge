from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.monitoring import completers
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.args import repeated
from googlecloudsdk.core.util import times
def AddPolicySettingsFlags(parser, update=False):
    """Adds policy settings flags to the parser."""
    policy_settings_group = parser.add_group(help='      Policy Settings.\n      If any of these are specified, they will overwrite fields in the\n      `--policy` or `--policy-from-file` flags if specified.')
    AddDisplayNameFlag(policy_settings_group, resource='Alert Policy')
    AddCombinerFlag(policy_settings_group, resource='Alert Policy')
    enabled_kwargs = {'action': arg_parsers.StoreTrueFalseAction if update else 'store_true'}
    if not update:
        enabled_kwargs['default'] = True
    policy_settings_group.add_argument('--enabled', help='If the policy is enabled.', **enabled_kwargs)
    documentation_group = policy_settings_group.add_group(help='Documentation')
    documentation_group.add_argument('--documentation-format', default='text/markdown' if not update else None, help='The MIME type that should be used with `--documentation` or `--documentation-from-file`. Currently, only "text/markdown" is supported.')
    documentation_string_group = documentation_group.add_group(mutex=True)
    documentation_string_group.add_argument('--documentation', help='The documentation to be included with the policy.')
    documentation_string_group.add_argument('--documentation-from-file', type=arg_parsers.FileContents(), help='The path to a file containing the documentation to be included with the policy.')
    if update:
        repeated.AddPrimitiveArgs(policy_settings_group, 'Alert Policy', 'notification-channels', 'Notification Channels')
        AddUpdateLabelsFlags('user-labels', policy_settings_group, group_text='User Labels')
    else:
        AddCreateLabelsFlag(policy_settings_group, 'user-labels', 'policy')