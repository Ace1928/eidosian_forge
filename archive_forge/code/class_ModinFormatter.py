import datetime as dt
import logging
import platform
import threading
import time
import uuid
from enum import IntEnum
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional
import pandas
import psutil
import modin
from modin.config import LogFileSize, LogMemoryInterval, LogMode
class ModinFormatter(logging.Formatter):
    """Implement custom formatter to log at microsecond granularity."""

    def formatTime(self, record: logging.LogRecord, datefmt: Optional[str]=None) -> str:
        """
        Return the creation time of the specified LogRecord as formatted text.

        This custom logging formatter inherits from the logging module and
        records timestamps at the microsecond level of granularity.

        Parameters
        ----------
        record : LogRecord
            The specified LogRecord object.
        datefmt : str, default: None
            Used with time.ststrftime() to format time record.

        Returns
        -------
        str
            Datetime string containing microsecond timestamp.
        """
        ct = dt.datetime.fromtimestamp(record.created)
        if datefmt:
            s = ct.strftime(datefmt)
        else:
            t = ct.strftime('%Y-%m-%d %H:%M:%S')
            s = f'{t},{record.msecs:03}'
        return s