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
def _dictify_context(context):
    if getattr(context, 'get_logging_values', None):
        return context.get_logging_values()
    elif getattr(context, 'to_dict', None):
        debtcollector.deprecate('The RequestContext.get_logging_values() method should be defined for logging context specific information.  The to_dict() method is deprecated for oslo.log use.', version='3.8.0', removal_version='5.0.0')
        return context.to_dict()
    elif isinstance(context, dict):
        return context
    return {}