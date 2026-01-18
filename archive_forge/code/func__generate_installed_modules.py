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
def _generate_installed_modules():
    try:
        from importlib import metadata
        yielded = set()
        for dist in metadata.distributions():
            name = dist.metadata['Name']
            if name is not None:
                normalized_name = _normalize_module_name(name)
                if dist.version is not None and normalized_name not in yielded:
                    yield (normalized_name, dist.version)
                    yielded.add(normalized_name)
    except ImportError:
        try:
            import pkg_resources
        except ImportError:
            return
        for info in pkg_resources.working_set:
            yield (_normalize_module_name(info.key), info.version)