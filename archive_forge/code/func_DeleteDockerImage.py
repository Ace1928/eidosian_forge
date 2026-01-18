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
def DeleteDockerImage(args):
    """Deletes a Docker digest or image.

  If input is an image, delete the image along with its resources.

  If input is an image identified by digest, delete the digest.
  If input is an image identified by tag, delete the digest and the tag.
  If --delete-tags is specified, delete all tags associated with the image
  digest.

  Args:
    args: user input arguments.

  Returns:
    The long-running operation from DeletePackage API call.
  """
    image, version_or_tag = _ParseDockerImage(args.IMAGE, _INVALID_IMAGE_ERROR)
    _ValidateDockerRepo(image.docker_repo.GetRepositoryName())
    client = ar_requests.GetClient()
    messages = ar_requests.GetMessages()
    if not version_or_tag:
        console_io.PromptContinue(message='\nThis operation will delete all tags and images for ' + image.GetDockerString() + '.', cancel_on_no=True)
        return ar_requests.DeletePackage(client, messages, image.GetPackageName())
    else:
        provided_tags = []
        docker_version = version_or_tag
        if isinstance(version_or_tag, DockerTag):
            docker_version = DockerVersion(version_or_tag.image, ar_requests.GetVersionFromTag(client, messages, version_or_tag.GetTagName()))
            provided_tags.append(version_or_tag)
        existing_tags = _GetDockerVersionTags(client, messages, docker_version)
        if not args.delete_tags and existing_tags != provided_tags:
            raise ar_exceptions.ArtifactRegistryError('Cannot delete image {} because it is tagged. Existing tags are:\n- {}'.format(args.IMAGE, '\n- '.join((tag.GetDockerString() for tag in existing_tags))))
        _LogResourcesToDelete(docker_version, existing_tags)
        console_io.PromptContinue(message='\nThis operation will delete the above resources.', cancel_on_no=True)
        for tag in existing_tags:
            ar_requests.DeleteTag(client, messages, tag.GetTagName())
        return ar_requests.DeleteVersion(client, messages, docker_version.GetVersionName())