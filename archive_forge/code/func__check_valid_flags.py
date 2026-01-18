from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import io
import re
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.document_renderers import text_renderer
import six
def _check_valid_flags(self, flags):
    for flag in flags:
        if flag not in self.command_metadata.flags:
            self.nonexistent_violation_flags.append(flag)