import datetime
import numbers
import re
import sys
import os
import textwrap
from tornado.escape import _unicode, native_str
from tornado.log import define_logging_options
from tornado.util import basestring_type, exec_in
from typing import (
def _help_callback(self, value: bool) -> None:
    if value:
        self.print_help()
        sys.exit(0)