from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.source import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddOptionalTopicFlags(group, resource_name='repo'):
    """Add message_format and service_account flags to the topic arg group."""
    group.add_argument('--message-format', choices=['json', 'protobuf'], help='The format of the message to publish to the topic.')
    group.add_argument('--service-account', help='Email address of the service account used for publishing Cloud Pub/Sub messages.\nThis service account needs to be in the same project as the {}. When added, the\ncaller needs to have iam.serviceAccounts.actAs permission on this service\naccount. If unspecified, it defaults to the Compute Engine default service\naccount.'.format(resource_name))
    group.add_argument('--topic-project', help='Cloud project for the topic. If not set, the currently set project will be\nused.')