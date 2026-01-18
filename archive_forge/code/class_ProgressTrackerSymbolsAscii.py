from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import locale
import os
import sys
import unicodedata
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr_os
from googlecloudsdk.core.console.style import text
from googlecloudsdk.core.util import encoding as encoding_util
import six
class ProgressTrackerSymbolsAscii(ProgressTrackerSymbols):
    """Characters used by progress trackers."""

    @property
    def spin_marks(self):
        return ['|', '/', '-', '\\']
    success = 'OK'
    failed = 'X'
    interrupted = '-'
    not_started = '.'
    prefix_length = 3