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
def _IsRequiredArg(self, attribute):
    """Returns True if the attribute arg should be required."""
    anchors = self._GetAnchors()
    return anchors == [attribute] and (not self.fallthroughs_map.get(attribute.name, []))