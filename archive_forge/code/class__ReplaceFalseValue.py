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
class _ReplaceFalseValue(dict):

    def __getitem__(self, key):
        return dict.get(self, key, None) or '-'