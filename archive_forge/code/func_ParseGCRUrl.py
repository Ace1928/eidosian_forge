from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import hashlib
import json
import re
from googlecloudsdk.api_lib.artifacts import exceptions as ar_exceptions
from googlecloudsdk.api_lib.container.images import util as gcr_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.artifacts import docker_util
from googlecloudsdk.core import log
from  googlecloudsdk.core.util.files import FileReader
def ParseGCRUrl(url):
    """Parse GCR URL.

  Args:
    url: gcr url for version, tag or whole image

  Returns:
    strings of project, image url and version url

  Raises:
    ar_exceptions.InvalidInputValueError: If user input is invalid.
  """
    location_map = {'us.gcr.io': 'us', 'gcr.io': 'us', 'eu.gcr.io': 'europe', 'asia.gcr.io': 'asia'}
    location = None
    project = None
    image = None
    matches = re.match(docker_util.GCR_DOCKER_REPO_REGEX, url)
    if matches:
        location = location_map[matches.group('repo')]
        project = matches.group('project')
        image = matches.group('image')
    matches = re.match(docker_util.GCR_DOCKER_DOMAIN_SCOPED_REPO_REGEX, url)
    if matches:
        location = location_map[matches.group('repo')]
        project = matches.group('project').replace('/', ':', 1)
        image = matches.group('image')
    if not project or not location or (not image):
        raise ar_exceptions.InvalidInputValueError('Failed to parse the GCR image.')
    matches = re.match(WHOLE_IMAGE_REGEX, image)
    if matches:
        return (project, url, None)
    try:
        docker_digest = gcr_util.GetDigestFromName(url)
    except gcr_util.InvalidImageNameError as e:
        raise ar_exceptions.InvalidInputValueError('Failed to resolve digest of the GCR image') from e
    image_url = super(type(docker_digest), docker_digest).__str__()
    return (project, image_url, str(docker_digest))