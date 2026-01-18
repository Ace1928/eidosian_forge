import base64
import json
import linecache
import logging
import math
import os
import random
import re
import subprocess
import sys
import threading
import time
from collections import namedtuple
from copy import copy
from decimal import Decimal
from numbers import Real
from datetime import datetime
from functools import partial
import sentry_sdk
from sentry_sdk._compat import PY2, PY33, PY37, implements_str, text_type, urlparse
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.consts import DEFAULT_MAX_VALUE_LENGTH
class AnnotatedValue(object):
    """
    Meta information for a data field in the event payload.
    This is to tell Relay that we have tampered with the fields value.
    See:
    https://github.com/getsentry/relay/blob/be12cd49a0f06ea932ed9b9f93a655de5d6ad6d1/relay-general/src/types/meta.rs#L407-L423
    """
    __slots__ = ('value', 'metadata')

    def __init__(self, value, metadata):
        self.value = value
        self.metadata = metadata

    def __eq__(self, other):
        if not isinstance(other, AnnotatedValue):
            return False
        return self.value == other.value and self.metadata == other.metadata

    @classmethod
    def removed_because_raw_data(cls):
        """The value was removed because it could not be parsed. This is done for request body values that are not json nor a form."""
        return AnnotatedValue(value='', metadata={'rem': [['!raw', 'x']]})

    @classmethod
    def removed_because_over_size_limit(cls):
        """The actual value was removed because the size of the field exceeded the configured maximum size (specified with the max_request_body_size sdk option)"""
        return AnnotatedValue(value='', metadata={'rem': [['!config', 'x']]})

    @classmethod
    def substituted_because_contains_sensitive_data(cls):
        """The actual value was removed because it contained sensitive information."""
        return AnnotatedValue(value=SENSITIVE_DATA_SUBSTITUTE, metadata={'rem': [['!config', 's']]})