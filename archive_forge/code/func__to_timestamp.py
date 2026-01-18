from __future__ import division
import datetime
import json
import logging
import os
import tempfile
from . import base
from ..discovery_cache import DISCOVERY_DOC_MAX_AGE
def _to_timestamp(date):
    try:
        return (date - EPOCH).total_seconds()
    except AttributeError:
        delta = date - EPOCH
        return (delta.microseconds + (delta.seconds + delta.days * 24 * 3600) * 10 ** 6) / 10 ** 6