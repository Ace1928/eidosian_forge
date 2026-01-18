from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps as deps_lib
from googlecloudsdk.calliope.concepts import deps_map_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.console import console_io
def Pluralize(self, attribute, plural=False):
    return plural and self.IsLeafAnchor(attribute)