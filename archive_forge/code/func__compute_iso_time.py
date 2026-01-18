import datetime
import debtcollector
import functools
import io
import itertools
import logging
import logging.config
import logging.handlers
import re
import socket
import sys
import traceback
from dateutil import tz
from oslo_context import context as context_utils
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
def _compute_iso_time(self, record):
    localtz = tz.tzlocal()
    record.isotime = datetime.datetime.fromtimestamp(record.created).replace(tzinfo=localtz).isoformat()
    if record.created == int(record.created):
        record.isotime = '%s.000000%s' % (record.isotime[:-6], record.isotime[-6:])