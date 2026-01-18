from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import datetime
import re
from googlecloudsdk.command_lib.ai import errors
from googlecloudsdk.command_lib.ai.custom_jobs import local_util
from googlecloudsdk.core import log
def GenerateImageName(base_name=None, project=None, region=None, is_gcr=False):
    """Generate a name for the Docker image built by AI platform gcloud."""
    sanitized_name = _SanitizeRepositoryName(base_name or _DEFAULT_IMAGE_NAME)
    tag = datetime.datetime.now().strftime('%Y%m%d.%H.%M.%S.%f')
    image_name = '{}/{}:{}'.format(_AUTONAME_PREFIX, sanitized_name, tag)
    if project:
        if is_gcr:
            repository = 'gcr.io'
        else:
            region_prefix = region or _DEFAULT_REPO_REGION
            repository = '{}-docker.pkg.dev'.format(region_prefix)
        return '{}/{}/{}'.format(repository, project.replace(':', '/'), image_name)
    return image_name