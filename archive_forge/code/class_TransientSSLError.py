import ssl
import time
import socket
import logging
from datetime import datetime, timedelta
from functools import wraps
from libcloud.utils.py3 import httplib
from libcloud.common.exceptions import RateLimitReachedError
class TransientSSLError(ssl.SSLError):
    """Represent transient SSL errors, e.g. timeouts"""
    pass