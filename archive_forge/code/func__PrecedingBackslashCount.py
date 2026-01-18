from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import io
import re
import textwrap
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import usage_text
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
import six
def _PrecedingBackslashCount(res):
    index = len(res) - 1
    while index >= 0 and res[index] == '\\':
        index -= 1
    return len(res) - index - 1