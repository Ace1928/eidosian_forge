from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import json
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.metastore import operations_util
from googlecloudsdk.api_lib.metastore import services_util as services_api_util
from googlecloudsdk.api_lib.metastore import util as api_util
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.metastore import resource_args
from googlecloudsdk.core import log
from googlecloudsdk.core.resource import resource_printer
import six
def ExtractQueryFolderUri(self, gcs_uri):
    """Returns the folder of query result gcs_uri.

    This takes gcs_uri and alter the filename to /filename[0]
    filename[0] is a string populated by grpc server.
      e.g., given gs://bucket-id/query-results/uuid/result-manifest
      output gs://bucket-id/query-results/uuid//

    Args:
      gcs_uri: the query metadata result gcs uri.
    """
    return gcs_uri[:gcs_uri.rfind('/')] + '//'