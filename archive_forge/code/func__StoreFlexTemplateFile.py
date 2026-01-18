from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
import shutil
import textwrap
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import exceptions
from googlecloudsdk.command_lib.builds import submit_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
import six
@staticmethod
def _StoreFlexTemplateFile(template_file_gcs_location, container_spec_json):
    """Stores flex template container spec file in GCS.

    Args:
      template_file_gcs_location: GCS location to store the template file.
      container_spec_json: Container spec in json format.

    Returns:
      Returns the stored flex template file gcs object on success.
      Propagates the error on failures.
    """
    with files.TemporaryDirectory() as temp_dir:
        local_path = os.path.join(temp_dir, 'template-file.json')
        files.WriteFileContents(local_path, container_spec_json, newline='\n')
        storage_client = storage_api.StorageClient()
        obj_ref = storage_util.ObjectReference.FromUrl(template_file_gcs_location)
        return storage_client.CopyFileToGCS(local_path, obj_ref)