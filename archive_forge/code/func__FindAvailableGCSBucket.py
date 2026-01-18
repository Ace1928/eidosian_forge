from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import hashlib
import json
import random
import re
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from containerregistry.client import docker_creds
from containerregistry.client import docker_name
from containerregistry.client.v2_2 import docker_http as v2_2_docker_http
from containerregistry.client.v2_2 import docker_image as v2_2_image
from containerregistry.client.v2_2 import docker_image_list as v2_2_image_list
from googlecloudsdk.api_lib.artifacts import exceptions as ar_exceptions
from googlecloudsdk.api_lib.cloudkms import base as cloudkms_base
from googlecloudsdk.api_lib.container.images import util as gcr_util
from googlecloudsdk.api_lib.containeranalysis import filter_util
from googlecloudsdk.api_lib.containeranalysis import requests as ca_requests
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.command_lib.artifacts import docker_util
from googlecloudsdk.command_lib.artifacts import requests as ar_requests
from googlecloudsdk.command_lib.artifacts import util
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core import transports
from googlecloudsdk.core.util import files
import requests
import six
from six.moves import urllib
def _FindAvailableGCSBucket(default_bucket, project_id, location):
    """Find an appropriate default bucket to store the SBOM file.

  Find a bucket with the same prefix same as the default bucket in the project.
  If no bucket could be found, will start to create a new bucket by
  concatenating the default bucket name and a random suffix.

  Args:
    default_bucket: str, targeting default bucket name for the resource.
    project_id: str, project we will use to store the SBOM.
    location: str, location we will use to store the SBOM.

  Returns:
    bucket_name: str, name of the prepared bucket.
  """
    gcs_client = storage_api.StorageClient()
    buckets = gcs_client.ListBuckets(project=project_id)
    for bucket in buckets:
        log.debug('Verifying bucket {}'.format(bucket.name))
        if not bucket.name.startswith(default_bucket):
            continue
        if bucket.locationType.lower() == 'dual-region':
            log.debug('Skipping dual region bucket {}'.format(bucket.name))
            continue
        if bucket.location.lower() != location.lower():
            log.debug('The bucket {0} has location {1} is not matching {2}.'.format(bucket.name, bucket.location.lower(), location.lower()))
            continue
        return bucket.name
    bucket_name = default_bucket + '-'
    for _ in range(_BUCKET_SUFFIX_LENGTH):
        bucket_name = bucket_name + random.choice(_BUCKET_NAME_CHARS)
    gcs_client.CreateBucketIfNotExists(bucket=bucket_name, project=project_id, location=location, check_ownership=True, enable_uniform_level_access=True)
    return bucket_name