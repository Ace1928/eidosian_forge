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
def _json_dumps_with_fallback(obj):
    convert = functools.partial(jsonutils.to_primitive, fallback=repr)
    return jsonutils.dumps(obj, default=convert)