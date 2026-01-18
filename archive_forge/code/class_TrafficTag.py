from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import operator
from googlecloudsdk.api_lib.kuberun import service
from googlecloudsdk.api_lib.kuberun import traffic
import six
class TrafficTag(object):
    """Contains the spec and status state for a traffic tag.

  Attributes:
    tag: The name of the tag.
    url: The tag's URL, or an empty string if the tag does not have a URL
      assigned yet. Defaults to an empty string.
    inSpec: Boolean that is true if the tag is present in the spec. Defaults to
      False.
    inStatus: Boolean that is true if the tag is present in the status. Defaults
      to False.
  """

    def __init__(self, tag, url='', in_spec=False, in_status=False):
        """Returns a new TrafficTag.

    Args:
      tag: The name of the tag.
      url: The tag's URL.
      in_spec: Boolean that is true if the tag is present in the spec.
      in_status: Boolean that is true if the tag is present in the status.
    """
        self.tag = tag
        self.url = url
        self.inSpec = in_spec
        self.inStatus = in_status