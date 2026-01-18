from __future__ import annotations
import email.utils
import re
import typing as t
import warnings
from datetime import date
from datetime import datetime
from datetime import time
from datetime import timedelta
from datetime import timezone
from enum import Enum
from hashlib import sha1
from time import mktime
from time import struct_time
from urllib.parse import quote
from urllib.parse import unquote
from urllib.request import parse_http_list as _parse_list_header
from ._internal import _dt_as_utc
from ._internal import _plain_int
from . import datastructures as ds
from .sansio import http as _sansio_http
class COOP(Enum):
    """Cross Origin Opener Policies"""
    UNSAFE_NONE = 'unsafe-none'
    SAME_ORIGIN_ALLOW_POPUPS = 'same-origin-allow-popups'
    SAME_ORIGIN = 'same-origin'