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
def _CreateTFBlueprint(messages, deployment_short_name, preview_short_name, location, local_source, stage_bucket, ignore_file, gcs_source, git_source_repo, git_source_directory, git_source_ref, input_values):
    """Returns the TerraformBlueprint message.

  Args:
    messages: ModuleType, the messages module that lets us form Config API
      messages based on our protos.
    deployment_short_name: short name of the deployment.
    preview_short_name: short name of the preview.
    location: location of the deployment.
    local_source: Local storage path where config files are stored.
    stage_bucket: optional string. Destination for storing local config files
      specified by local source flag. e.g. "gs://bucket-name/".
    ignore_file: optional string, a path to a gcloudignore file.
    gcs_source:  URI of an object in Google Cloud Storage. e.g.
      `gs://{bucket}/{object}`
    git_source_repo: Repository URL.
    git_source_directory: Subdirectory inside the git repository.
    git_source_ref: Git branch or tag.
    input_values: Input variable values for the Terraform blueprint. It only
      accepts (key, value) pairs where value is a scalar value.

  Returns:
    A messages.TerraformBlueprint to use with deployment operation.
  """
    terraform_blueprint = messages.TerraformBlueprint(inputValues=input_values)
    if gcs_source is not None:
        terraform_blueprint.gcsSource = gcs_source
    elif local_source is not None and deployment_short_name is not None:
        upload_bucket = _UploadSourceToGCS(local_source, stage_bucket, deployment_short_name, location, ignore_file)
        terraform_blueprint.gcsSource = upload_bucket
    elif local_source is not None and preview_short_name is not None:
        upload_bucket = _UploadSourceToGCS(local_source, stage_bucket, preview_short_name, location, ignore_file)
        terraform_blueprint.gcsSource = upload_bucket
    else:
        terraform_blueprint.gitSource = messages.GitSource(repo=git_source_repo, directory=git_source_directory, ref=git_source_ref)
    return terraform_blueprint