from __future__ import division
import datetime
import json
import logging
import os
import tempfile
from . import base
from ..discovery_cache import DISCOVERY_DOC_MAX_AGE
Constructor.

        Args:
          max_age: Cache expiration in seconds.
        