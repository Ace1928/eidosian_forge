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
def _CreatePreviewOp(preview, preview_full_name, location_full_name):
    """Initiates and returns a CreatePreview operation.

  Args:
    preview: A partially filled messages.preview. The preview will be filled
      with other details before the operation is initiated.
    preview_full_name: string, the fully qualified name of the preview, e.g.
      "projects/p/locations/l/previews/p".
    location_full_name: string, the fully qualified name of the location, e.g.
      "projects/p/locations/l".

  Returns:
    The CreatePreview operation.
  """
    if preview_full_name is None:
        return configmanager_util.CreatePreview(preview, None, location_full_name)
    preview_ref = resources.REGISTRY.Parse(preview_full_name, collection='config.projects.locations.previews')
    preview_id = preview_ref.Name()
    log.info('Creating the preview')
    return configmanager_util.CreatePreview(preview, preview_id, location_full_name)