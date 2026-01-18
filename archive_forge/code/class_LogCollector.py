import logging
import unittest
import weakref
from typing import Dict, List
from .. import pyutils
class LogCollector(logging.Handler):

    def __init__(self):
        logging.Handler.__init__(self)
        self.records = []

    def emit(self, record):
        self.records.append(record.getMessage())