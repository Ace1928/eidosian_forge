from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import io
import re
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.document_renderers import text_renderer
import six
def _analyze_example_flags_equals(self, flags):
    for flag in flags:
        if '=' not in flag and flag not in self.command_metadata.bool_flags and (flag not in self._NON_BOOL_FLAGS_ALLOWLIST):
            self.equals_violation_flags.append(flag)