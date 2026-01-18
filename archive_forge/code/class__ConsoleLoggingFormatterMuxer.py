from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from collections import OrderedDict
import contextlib
import copy
import datetime
import json
import logging
import os
import sys
import time
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console.style import parser as style_parser
from googlecloudsdk.core.console.style import text
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
class _ConsoleLoggingFormatterMuxer(logging.Formatter):
    """Logging Formatter Composed of other formatters."""

    def __init__(self, structured_formatter, stream_writter, default_formatter=None):
        logging.Formatter.__init__(self)
        self.default_formatter = default_formatter or logging.Formatter
        self.structured_formatter = structured_formatter
        self.terminal = stream_writter.isatty()

    def ShowStructuredOutput(self):
        """Returns True if output should be Structured, False otherwise."""
        show_messages = properties.VALUES.core.show_structured_logs.Get()
        if any([show_messages == 'terminal' and self.terminal, show_messages == 'log' and (not self.terminal), show_messages == 'always']):
            return True
        return False

    def format(self, record):
        """Formats the record using the proper formatter."""
        show_structured_output = self.ShowStructuredOutput()
        stylize = self.terminal and (not show_structured_output)
        record = copy.copy(record)
        if isinstance(record.msg, text.TypedText):
            record.msg = style_parser.GetTypedTextParser().ParseTypedTextToString(record.msg, stylize=stylize)
        if isinstance(record.args, tuple):
            new_args = []
            for arg in record.args:
                if isinstance(arg, text.TypedText):
                    arg = style_parser.GetTypedTextParser().ParseTypedTextToString(arg, stylize=stylize)
                new_args.append(arg)
            record.args = tuple(new_args)
        if show_structured_output:
            return self.structured_formatter.format(record)
        return self.default_formatter.format(record)