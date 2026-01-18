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
class _JsonFormatter(logging.Formatter):
    """A formatter that handles formatting log messages as JSON."""

    def __init__(self, required_fields, json_serializer=None, json_encoder=None):
        super(_JsonFormatter, self).__init__()
        self.required_fields = required_fields
        self.json_encoder = json_encoder
        self.json_serializer = json_serializer or json.dumps
        self.default_time_format = STRUCTURED_TIME_FORMAT

    def GetErrorDict(self, log_record):
        """Extract exception info from a logging.LogRecord as an OrderedDict."""
        error_dict = OrderedDict()
        if log_record.exc_info:
            if not log_record.exc_text:
                log_record.exc_text = self.formatException(log_record.exc_info)
            if issubclass(type(log_record.msg), BaseException):
                error_dict['type'] = type(log_record.msg).__name__
                error_dict['details'] = six.text_type(log_record.msg)
                error_dict['stacktrace'] = getattr(log_record.msg, '__traceback__', None)
            elif issubclass(type(log_record.exc_info[0]), BaseException):
                error_dict['type'] = log_record.exc_info[0]
                error_dict['details'] = log_record.exc_text
                error_dict['stacktrace'] = log_record.exc_info[2]
            else:
                error_dict['type'] = log_record.exc_text
                error_dict['details'] = log_record.exc_text
                error_dict['stacktrace'] = log_record.exc_text
            return error_dict
        return None

    def BuildLogMsg(self, log_record):
        """Converts a logging.LogRecord object to a JSON serializable OrderedDict.

    Utilizes supplied set of required_fields to determine output fields.

    Args:
      log_record: logging.LogRecord, log record to be converted

    Returns:
      OrderedDict of required_field values.
    """
        message_dict = OrderedDict()
        for outfield, logfield in six.iteritems(self.required_fields):
            if outfield == 'version':
                message_dict[outfield] = STRUCTURED_RECORD_VERSION
            else:
                message_dict[outfield] = log_record.__dict__.get(logfield)
        return message_dict

    def LogRecordToJson(self, log_record):
        """Returns a json string of the log message."""
        log_message = self.BuildLogMsg(log_record)
        if not log_message.get('error'):
            log_message.pop('error')
        return self.json_serializer(log_message, cls=self.json_encoder)

    def formatTime(self, record, datefmt=None):
        return times.FormatDateTime(times.GetDateTimeFromTimeStamp(record.created), fmt=datefmt, tzinfo=times.UTC)

    def format(self, record):
        """Formats a log record and serializes to json."""
        record.__dict__['error'] = self.GetErrorDict(record)
        record.message = record.getMessage()
        record.asctime = self.formatTime(record, self.default_time_format)
        return self.LogRecordToJson(record)