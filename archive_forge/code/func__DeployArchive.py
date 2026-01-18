from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib import apigee
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.apigee import archives as cmd_lib
from googlecloudsdk.command_lib.apigee import defaults
from googlecloudsdk.command_lib.apigee import errors
from googlecloudsdk.command_lib.apigee import resource_args
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
def _DeployArchive(self, identifiers, upload_url, labels):
    """Creates the archive deployment.

    Args:
      identifiers: A dict of resource identifers. Must contain "organizationsId"
        and "environmentsId"
      upload_url: A str containing the full upload URL.
      labels: A dict of the key/value pairs to add as labels.

    Returns:
      A dict containing the operation metadata.
    """
    post_data = {}
    post_data['gcs_uri'] = upload_url
    if labels:
        post_data['labels'] = {}
        for k, v in labels.items():
            post_data['labels'][k] = v
    api_response = apigee.ArchivesClient.CreateArchiveDeployment(identifiers, post_data)
    operation = apigee.OperationsClient.SplitName(api_response)
    return operation