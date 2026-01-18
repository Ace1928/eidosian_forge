from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import datetime
import re
from googlecloudsdk.command_lib.ai import errors
from googlecloudsdk.command_lib.ai.custom_jobs import local_util
from googlecloudsdk.core import log
def ValidateRepositoryAndTag(image_name):
    """Validate the given image name is a valid repository/tag reference.

  As explained in
  https://docs.docker.com/engine/reference/commandline/tag/#extended-description,
  a valid repository/tag reference should following the below pattern:

  reference             := name [ ":" tag ]
  name                  := [hostname '/'] component ['/' component]*
  hostname              := hostcomponent ['.' hostcomponent]* [':' port-number]
  hostcomponent         := /([a-zA-Z0-9]|[a-zA-Z0-9][a-zA-Z0-9-]*[a-zA-Z0-9])/
  port-number           := /[0-9]+/
  component             := alpha-numeric [separator alpha-numeric]*
  alpha-numeric         := /[a-z0-9]+/
  separator             := /[_.]|__|[-]*/

  tag                   := /[\\w][\\w.-]{0,127}/

  Args:
    image_name: (str) Full name of a Docker image.

  Raises:
    ValueError if the image name is not valid.
  """
    repository, tag = _ParseRepositoryTag(image_name)
    if repository is None:
        raise ValueError('Unable to parse repository and tag.')
    if len(repository) > _MAX_REPOSITORY_LENGTH:
        raise ValueError('Repository name must not be more than {} characters.'.format(_MAX_REPOSITORY_LENGTH))
    hostname, path_components = _ParseRepositoryHost(repository)
    if hostname:
        hostcomponents, port = _ParseHostPort(hostname)
        hostcomponent_regex = '^(?:[a-zA-Z0-9]|[a-zA-Z0-9][a-zA-Z0-9-]*[a-zA-Z0-9])$'
        for hostcomponent in hostcomponents:
            if re.match(hostcomponent_regex, hostcomponent) is None:
                raise ValueError('Invalid hostname/port "{}" in repository name.'.format(hostname))
        port_regex = '^[0-9]+$'
        if port and re.match(port_regex, port) is None:
            raise ValueError('Invalid hostname/port "{}" in repository name.'.format(hostname))
    for component in path_components:
        if not component:
            raise ValueError('Empty path component in repository name.')
        component_regex = '^[a-z0-9]+(?:(?:[._]|__|[-]*)[a-z0-9]+)*$'
        if re.match(component_regex, component) is None:
            raise ValueError('Invalid path component "{}" in repository name.'.format(component))
    if tag:
        if len(tag) > _MAX_TAG_LENGTH:
            raise ValueError('Tag name must not be more than {} characters.'.format(_MAX_TAG_LENGTH))
        tag_regex = '^[\\w][\\w.-]{0,127}$'
        if re.match(tag_regex, tag) is None:
            raise ValueError('Invalid tag.')