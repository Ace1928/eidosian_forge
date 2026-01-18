from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from apitools.base.py import extra_types
from googlecloudsdk.api_lib.resource_manager import folders
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.resource_manager import completers
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import common_args
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log as sdk_log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
def PrintPermissionInstructions(destination, writer_identity):
    """Prints a message to remind the user to set up permissions for a sink.

  Args:
    destination: the sink destination (either bigquery or cloud storage).
    writer_identity: identity to which to grant write access.
  """
    if writer_identity:
        grantee = '`{0}`'.format(writer_identity)
    else:
        grantee = 'the group `cloud-logs@google.com`'
    if destination.startswith('bigquery'):
        sdk_log.status.Print('Please remember to grant {0} the BigQuery Data Editor role on the dataset.'.format(grantee))
    elif destination.startswith('storage'):
        sdk_log.status.Print('Please remember to grant {0} the Storage Object Creator role on the bucket.'.format(grantee))
    elif destination.startswith('pubsub'):
        sdk_log.status.Print('Please remember to grant {0} the Pub/Sub Publisher role on the topic.'.format(grantee))
    sdk_log.status.Print('More information about sinks can be found at https://cloud.google.com/logging/docs/export/configure_export')