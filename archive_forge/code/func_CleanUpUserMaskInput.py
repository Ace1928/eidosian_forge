from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import re
from googlecloudsdk.command_lib.scc import errors
from googlecloudsdk.core import properties
def CleanUpUserMaskInput(mask):
    """Removes spaces from a field mask provided by user."""
    return mask.replace(' ', '')