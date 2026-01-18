from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding as apitools_encoding
from googlecloudsdk.api_lib.container.fleet import client as hub_client
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.resource import resource_transform
def TransformBuildImages(r, undefined=''):
    """Returns the formatted build results images.

  Args:
    r: JSON-serializable object.
    undefined: Returns this value if the resource cannot be formatted.
  Returns:
    The formatted build results images.
  """
    messages = core_apis.GetMessagesModule('cloudbuild', 'v1')
    b = apitools_encoding.DictToMessage(r, messages.Build)
    if b.results is None:
        return undefined
    images = b.results.images
    if not images:
        return undefined
    names = []
    for i in images:
        if i.name is None:
            names.append(undefined)
        else:
            names.append(i.name)
    if len(names) > 1:
        return names[0] + ' (+{0} more)'.format(len(names) - 1)
    return names[0]