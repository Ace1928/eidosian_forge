import datetime
import functools
import hashlib
import json
import logging
import os
import platform
import socket
import sys
import time
import urllib
import uuid
import requests
import keystoneauth1
from keystoneauth1 import _utils as utils
from keystoneauth1 import discover
from keystoneauth1 import exceptions
class RequestTiming(object):
    """Contains timing information for an HTTP interaction."""
    method = None
    url = None
    elapsed = None

    def __init__(self, method, url, elapsed):
        self.method = method
        self.url = url
        self.elapsed = elapsed