from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import io
import re
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.document_renderers import text_renderer
import six
def _CaptureOutput(self, heading):
    self.check_indentation_for_examples()
    if self._buffer.getvalue() and self._prev_heading:
        self._Analyze(self._prev_heading, self._buffer.getvalue())
        self._buffer = io.StringIO()
    if self._prev_heading == 'EXAMPLES':
        self.check_example_section_errors()
    self._out = self._buffer
    self._prev_heading = self._heading