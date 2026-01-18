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
@contextlib.contextmanager
def _SafeDecodedLogRecord(record, encoding):
    """Temporarily modifies a log record to make the message safe to print.

  Python logging creates a single log record for each log event. Each handler
  is given that record and asked format it. To avoid unicode issues, we decode
  all the messages in case they are byte strings. Doing this we also want to
  ensure the resulting string is able to be printed to the given output target.

  Some handlers target the console (which can have many different encodings) and
  some target the log file (which we always write as utf-8. If we modify the
  record, depending on the order of handlers, the log message could lose
  information along the way.

  For example, if the user has an ascii console, we replace non-ascii characters
  in the string with '?' to print. Then if the log file handler is called, the
  original unicode data is gone, even though it could successfully be printed
  to the log file. This context manager changes the log record briefly so it can
  be formatted without changing it for later handlers.

  Args:
    record: The log record.
    encoding: The name of the encoding to SafeDecode with.
  Yields:
    None, yield is necessary as this is a context manager.
  """
    original_msg = record.msg
    try:
        record.msg = console_attr.SafeText(record.msg, encoding=encoding, escape=False)
        yield
    finally:
        record.msg = original_msg