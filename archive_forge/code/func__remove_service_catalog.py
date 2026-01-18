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
def _remove_service_catalog(self, body):
    try:
        data = json.loads(body)
        if 'token' in data and 'catalog' in data['token']:
            data['token']['catalog'] = '<removed>'
            return self._json.encode(data)
        if 'serviceCatalog' in data['access']:
            data['access']['serviceCatalog'] = '<removed>'
            return self._json.encode(data)
    except Exception:
        pass
    return body