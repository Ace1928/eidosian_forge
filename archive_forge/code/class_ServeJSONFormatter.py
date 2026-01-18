import copy
import json
import logging
import os
from typing import Optional, Tuple
import ray
from ray.serve._private.common import ServeComponentType
from ray.serve._private.constants import (
from ray.serve.schema import EncodingType, LoggingConfig
class ServeJSONFormatter(logging.Formatter):
    """Serve Logging Json Formatter

    The formatter will generate the json log format on the fly
    based on the field of record.
    """

    def __init__(self, component_name: str, component_id: str, component_type: Optional[ServeComponentType]=None):
        self.component_log_fmt = {SERVE_LOG_LEVEL_NAME: SERVE_LOG_RECORD_FORMAT[SERVE_LOG_LEVEL_NAME], SERVE_LOG_TIME: SERVE_LOG_RECORD_FORMAT[SERVE_LOG_TIME]}
        if component_type and component_type == ServeComponentType.REPLICA:
            self.component_log_fmt[SERVE_LOG_DEPLOYMENT] = component_name
            self.component_log_fmt[SERVE_LOG_REPLICA] = component_id
        else:
            self.component_log_fmt[SERVE_LOG_COMPONENT] = component_name
            self.component_log_fmt[SERVE_LOG_COMPONENT_ID] = component_id

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record into json format.

        Args:
            record: The log record to be formatted.

            Returns:
                The formatted log record in json format.
        """
        record_format = copy.deepcopy(self.component_log_fmt)
        if SERVE_LOG_REQUEST_ID in record.__dict__:
            record_format[SERVE_LOG_REQUEST_ID] = SERVE_LOG_RECORD_FORMAT[SERVE_LOG_REQUEST_ID]
        if SERVE_LOG_ROUTE in record.__dict__:
            record_format[SERVE_LOG_ROUTE] = SERVE_LOG_RECORD_FORMAT[SERVE_LOG_ROUTE]
        if SERVE_LOG_APPLICATION in record.__dict__:
            record_format[SERVE_LOG_APPLICATION] = SERVE_LOG_RECORD_FORMAT[SERVE_LOG_APPLICATION]
        message_formatter = logging.Formatter(SERVE_LOG_RECORD_FORMAT[SERVE_LOG_MESSAGE])
        record_format[SERVE_LOG_MESSAGE] = message_formatter.format(record)
        if SERVE_LOG_EXTRA_FIELDS in record.__dict__:
            if not isinstance(record.__dict__[SERVE_LOG_EXTRA_FIELDS], dict):
                raise ValueError(f'Expected a dictionary passing into {SERVE_LOG_EXTRA_FIELDS}, but got {type(record.__dict__[SERVE_LOG_EXTRA_FIELDS])}')
            for k, v in record.__dict__[SERVE_LOG_EXTRA_FIELDS].items():
                if k in record_format:
                    raise KeyError(f'Found duplicated key in the log record: {k}')
                record_format[k] = v
        formatter = logging.Formatter(json.dumps(record_format))
        return formatter.format(record)