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
def _sanitize_headers(headers):
    """Ensure headers are strings and not bytes."""
    str_dict = {}
    for k, v in headers.items():
        k = k.decode('ASCII') if isinstance(k, bytes) else k
        if v is not None:
            v = v.decode('ASCII') if isinstance(v, bytes) else v
        str_dict[k] = v
    return str_dict