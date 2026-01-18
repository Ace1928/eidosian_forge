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
class _LogFileFormatter(logging.Formatter):
    """A formatter for log file contents."""
    FORMAT = _FmtString('%(asctime)s %(levelname)-8s %(name)-15s %(message)s')

    def __init__(self):
        super(_LogFileFormatter, self).__init__(fmt=_LogFileFormatter.FORMAT)

    def format(self, record):
        record = copy.copy(record)
        if isinstance(record.msg, text.TypedText):
            record.msg = style_parser.GetTypedTextParser().ParseTypedTextToString(record.msg, stylize=False)
        if isinstance(record.args, tuple):
            new_args = []
            for arg in record.args:
                if isinstance(arg, text.TypedText):
                    arg = style_parser.GetTypedTextParser().ParseTypedTextToString(arg, stylize=False)
                new_args.append(arg)
            record.args = tuple(new_args)
        with _SafeDecodedLogRecord(record, LOG_FILE_ENCODING):
            msg = super(_LogFileFormatter, self).format(record)
        return msg