from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.components import util
from googlecloudsdk.core.updater import update_manager
Make a temporary copy of bundled Python installation.

  Also print its location.

  If the Python installation used to execute this command is *not* bundled, do
  not make a copy. Instead, print the location of the current Python
  installation.
  