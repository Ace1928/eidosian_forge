from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import io
import re
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.document_renderers import text_renderer
import six
def _add_no_errors_summary(self, heading):
    self.findings['There are no errors for the {} section.'.format(heading)] = ''