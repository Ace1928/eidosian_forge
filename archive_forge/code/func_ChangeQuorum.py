from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from cloudsdk.google.protobuf import descriptor_pb2
from cloudsdk.google.protobuf import text_format
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.ai import errors
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.spanner.resource_args import CloudKmsKeyName
def ChangeQuorum(database_ref, quorum_type, etag=None):
    """ChangeQuorum a database."""
    client = apis.GetClientInstance('spanner', 'v1')
    msgs = apis.GetMessagesModule('spanner', 'v1')
    req = msgs.ChangeQuorumRequest(etag=etag, name=database_ref.RelativeName(), quorumType=quorum_type)
    return client.projects_instances_databases.Changequorum(req)