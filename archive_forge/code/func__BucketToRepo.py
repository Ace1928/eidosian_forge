import dataclasses
from typing import Iterator
from apitools.base.py import list_pager
from containerregistry.client import docker_name
from containerregistry.client.v2_2 import docker_http
from containerregistry.client.v2_2 import docker_image
from googlecloudsdk.api_lib.asset import client_util as asset_client_util
from googlecloudsdk.api_lib.container.images import util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def _BucketToRepo(bucket: asset_client_util.GetMessages().ResourceSearchResult) -> docker_name.Repository:
    """Converts a GCS bucket to a GCR repo.

  Args:
    bucket: A CAIS ResourceSearchResult for a GCS bucket.

  Returns:
    A docker repository.
  """
    project_prefix = f'//cloudresourcemanager.{properties.VALUES.core.universe_domain.Get()}/projects/'
    if not bucket.parentFullResourceName.startswith(project_prefix):
        log.warning(f'{bucket.parentFullResourceName} is not a Project name. Skipping...')
        return None
    project_id = bucket.parentFullResourceName[len(project_prefix):]
    bucket_prefix = f'//storage.{properties.VALUES.core.universe_domain.Get()}/'
    bucket_suffix = _BucketSuffix(project_id)
    if not bucket.name.startswith(bucket_prefix) or not bucket.name.endswith(bucket_suffix):
        log.warning(f'{bucket.name} is not a Container Registry bucket. Skipping...')
        return None
    gcr_region_prefix = bucket.name[len(bucket_prefix):-len(bucket_suffix)]
    if gcr_region_prefix not in _VALID_GCR_REGION_PREFIX:
        log.warning(f'{bucket.name} is not a Container Registry bucket. Skipping...')
        return None
    gcr_repo_path = '{region}gcr.io/{project}'.format(region=gcr_region_prefix, project=project_id.replace(':', '/', 1))
    return util.ValidateRepositoryPath(gcr_repo_path)