from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import deps as deps_lib
from googlecloudsdk.calliope.concepts import util
from googlecloudsdk.command_lib.util.concepts import completers
from googlecloudsdk.core.util import text
import six
from six.moves import filter  # pylint: disable=redefined-builtin
def GetHints(self, attribute_name):
    """Gets a list of string hints for how to set an attribute.

    Given the attribute name, gets a list of hints corresponding to the
    attribute's fallthroughs.

    Args:
      attribute_name: str, the name of the attribute.

    Returns:
      A list of hints for its fallthroughs, including its primary arg if any.
    """
    fallthroughs = self.BuildFullFallthroughsMap().get(attribute_name, [])
    return deps_lib.GetHints(fallthroughs)