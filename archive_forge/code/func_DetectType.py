from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from os import path
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from googlecloudsdk.api_lib.data_catalog import entries_v1
from googlecloudsdk.api_lib.data_catalog import util as api_util
from googlecloudsdk.command_lib.concepts import exceptions as concept_exceptions
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
def DetectType(ref, args, request):
    """Detect Entry type.

  Args:
    ref: The entry resource reference.
    args: The parsed args namespace.
    request: The update entry request.

  Returns:
    Request with proper type
  """
    del ref
    client = entries_v1.EntriesClient()
    messages = client.messages
    if args.IsSpecified('kafka_cluster_bootstrap_servers'):
        arg_utils.SetFieldInMessage(request, 'googleCloudDatacatalogV1Entry.type', messages.GoogleCloudDatacatalogV1Entry.TypeValueValuesEnum.CLUSTER)
    if args.IsSpecified('kafka_topic'):
        arg_utils.SetFieldInMessage(request, 'googleCloudDatacatalogV1Entry.type', messages.GoogleCloudDatacatalogV1Entry.TypeValueValuesEnum.DATA_STREAM)
    return request