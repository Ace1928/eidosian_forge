from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import exceptions as api_exceptions
from googlecloudsdk.api_lib.artifacts import exceptions as ar_exceptions
from googlecloudsdk.api_lib.util import common_args
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.artifacts import containeranalysis_util as ca_util
from googlecloudsdk.command_lib.artifacts import requests as ar_requests
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
def _GetDockerPackagesAndVersions(docker_repo, include_tags, page_size, order_by, limit, package_prefix=''):
    """Gets a list of packages with versions for a Docker repository."""
    client = ar_requests.GetClient()
    messages = ar_requests.GetMessages()
    img_list = []
    for pkg in ar_requests.ListPackages(client, messages, docker_repo.GetRepositoryName(), page_size=page_size):
        parts = pkg.name.split('/')
        if len(parts) != 8:
            raise ar_exceptions.ArtifactRegistryError('Internal error. Corrupted package name: {}'.format(pkg.name))
        img = DockerImage(DockerRepo(parts[1], parts[3], parts[5]), parts[7])
        if package_prefix and (not img.GetDockerString().startswith(package_prefix)):
            continue
        img_list.extend(_GetDockerVersions(img, include_tags, page_size, order_by, limit, search_subdirs=False))
    return img_list