import base64
import datetime
import decimal
import inspect
import io
import logging
import netaddr
import re
import sys
import uuid
import weakref
from wsme import exc
def _set_content(self, value):
    self._content = value
    self._file = None