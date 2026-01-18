from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
import sys
from googlecloudsdk.command_lib.interactive import browser
from prompt_toolkit import keys
from prompt_toolkit.key_binding import manager
import six
class _StopKeyBinding(_KeyBinding):
    """The stop (^Z) key binding.

  This binding's sole purpose is to ignore ^Z and prevent it from echoing
  in the prompt window.
  """

    def __init__(self, key):
        super(_StopKeyBinding, self).__init__(key=key)