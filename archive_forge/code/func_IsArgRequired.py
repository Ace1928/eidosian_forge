from googlecloudsdk.command_lib.concepts import concept_managers
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import io
import re
from googlecloudsdk.calliope.concepts import deps as deps_lib
from googlecloudsdk.command_lib.concepts import base
from googlecloudsdk.command_lib.concepts import dependency_managers
from googlecloudsdk.command_lib.concepts import exceptions
from googlecloudsdk.command_lib.concepts import names
from googlecloudsdk.core.util import scaled_integer
from googlecloudsdk.core.util import semver
from googlecloudsdk.core.util import times
import six
def IsArgRequired(self):
    """Determines whether the concept group is required to be specified.

    Returns:
      bool: True, if the command line argument is required to be provided,
        meaning that the attribute is required and that there are no
        fallthroughs. There may still be a parsing error if the argument isn't
        provided and none of the fallthroughs work.
    """
    return self.required and (not any((c.fallthroughs for c in self.concepts)))