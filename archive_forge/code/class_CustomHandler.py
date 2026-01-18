import io
import logging
import time
import testtools
from testtools import TestCase
from fixtures import (
class CustomHandler(logging.Handler):

    def __init__(self, *args, **kwargs):
        """Create the instance, and add a records attribute."""
        logging.Handler.__init__(self, *args, **kwargs)
        self.msgs = []

    def emit(self, record):
        self.msgs.append(record.msg)