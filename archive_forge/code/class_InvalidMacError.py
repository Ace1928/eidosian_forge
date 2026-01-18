import base64
import functools
import hashlib
import hmac
import math
import os
from keystonemiddleware.i18n import _
from oslo_utils import secretutils
class InvalidMacError(Exception):
    """raise when unable to verify MACed data.

    This usually indicates that data had been expectedly modified in memcache.

    """
    pass