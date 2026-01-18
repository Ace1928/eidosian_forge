from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import sys
import uuid
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import extra_types
from googlecloudsdk.api_lib.infra_manager import configmanager_util
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.infra_manager import deterministic_snapshot
from googlecloudsdk.command_lib.infra_manager import errors
from googlecloudsdk.command_lib.infra_manager import staging_bucket_util
from googlecloudsdk.core import log
from googlecloudsdk.core import requests
from googlecloudsdk.core import resources
from googlecloudsdk.core.resource import resource_transform
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
def DeleteCleanupStagedObjects(response, unused_args):
    """DeploymentDeleteCleanupStagedObjects deletes staging gcs objects created as part of deployment apply command."""
    if response.error is not None:
        return
    if response.metadata is not None:
        md = encoding.MessageToPyValue(response.metadata)
        entity_full_name = md['target']
        entity_id = entity_full_name.split('/')[5]
        location = entity_full_name.split('/')[3]
        staging_gcs_directory = staging_bucket_util.DefaultGCSStagingDir(entity_id, location)
        gcs_client = storage_api.StorageClient()
        staging_bucket_util.DeleteStagingGCSFolder(gcs_client, staging_gcs_directory)
    return response