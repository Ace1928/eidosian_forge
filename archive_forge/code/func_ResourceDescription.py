from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import pkgutil
import textwrap
from googlecloudsdk import api_lib
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.resource import resource_transform
import six
from six.moves import range  # pylint: disable=redefined-builtin
def ResourceDescription(name):
    """Generates resource help DESCRIPTION help text for name.

  This puts common text for the key, formats and projections topics in one
  place.

  Args:
    name: One of ['filter', 'format', 'key', 'projection'].

  Raises:
    ValueError: If name is not one of the expected topic names.

  Returns:
    A detailed_help DESCRIPTION markdown string.
  """
    description = '  Most *gcloud* commands return a list of resources on success. By default they\n  are pretty-printed on the standard output. The\n  *--format=*_NAME_[_ATTRIBUTES_]*(*_PROJECTION_*)* and\n  *--filter=*_EXPRESSION_ flags along with projections can be used to format and\n  change the default output to a more meaningful result.\n\n  Use the `--format` flag to change the default output format of a command.   {see_format}\n\n  Use the `--filter` flag to select resources to be listed. {see_filter}\n\n  Use resource-keys to reach resource items through a unique path of names from the root. {see_key}\n\n  Use projections to list a subset of resource keys in a resource.   {see_projection}\n\n  Note: To refer to a list of fields you can sort, filter, and format by for\n  each resource, you can run a list command with the format set to `text` or\n  `json`. For\n  example, $ gcloud compute instances list --limit=1 --format=text.\n\n  To work through an interactive tutorial about using the filter and format\n  flags instead, see: https://console.cloud.google.com/cloudshell/open?git_repo=https://github.com/GoogleCloudPlatform/cloud-shell-tutorials&page=editor&tutorial=cloudsdk/tutorial.md\n  '
    topics = {'filter': 'filters', 'format': 'formats', 'key': 'resource-keys', 'projection': 'projections'}
    if name not in topics:
        raise ValueError('Expected one of [{topics}], got [{name}].'.format(topics=','.join(sorted(topics)), name=name))
    see = {}
    for topic, command in six.iteritems(topics):
        if topic == name:
            see[topic] = 'Resource {topic}s are described in detail below.'.format(topic=topic)
        else:
            see[topic] = 'For details run $ gcloud topic {command}.'.format(command=command)
    return textwrap.dedent(description).format(see_filter=see['filter'], see_format=see['format'], see_key=see['key'], see_projection=see['projection'])