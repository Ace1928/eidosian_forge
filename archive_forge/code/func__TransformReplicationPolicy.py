from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.secrets import args as secrets_args
def _TransformReplicationPolicy(r):
    if 'replication' not in r:
        return 'ERROR'
    if 'automatic' in r['replication']:
        return 'automatic'
    if 'userManaged' in r['replication']:
        return 'user_managed'
    return 'ERROR'