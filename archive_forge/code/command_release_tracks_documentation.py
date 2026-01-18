from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
from googlecloudsdk.calliope import base
Separate combined track definitions.

  If a file does not specify tracks, the same implementation may be used for
  all track implementations the command is present in.

  Args:
    command_impls: A single or list of declarative command implementation(s).
  Yields:
    One implementation for each distinct track implmentation in a file.
  