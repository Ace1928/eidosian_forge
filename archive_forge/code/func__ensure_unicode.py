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
def _ensure_unicode(msg):
    """Do our best to turn the input argument into a unicode object.
    """
    if isinstance(msg, str):
        return msg
    if not isinstance(msg, bytes):
        return str(msg)
    return encodeutils.safe_decode(msg, incoming='utf-8', errors='xmlcharrefreplace')