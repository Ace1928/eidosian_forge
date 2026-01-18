from functools import reduce
import gc
import io
import locale  # system locale module, not tornado.locale
import logging
import operator
import textwrap
import sys
import unittest
import warnings
from tornado.httpclient import AsyncHTTPClient
from tornado.httpserver import HTTPServer
from tornado.netutil import Resolver
from tornado.options import define, add_parse_callback, options
class LogCounter(logging.Filter):
    """Counts the number of WARNING or higher log records."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.info_count = self.warning_count = self.error_count = 0

    def filter(self, record):
        if record.levelno >= logging.ERROR:
            self.error_count += 1
        elif record.levelno >= logging.WARNING:
            self.warning_count += 1
        elif record.levelno >= logging.INFO:
            self.info_count += 1
        return True