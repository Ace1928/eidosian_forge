import abc
import contextlib
import copy
import hashlib
import os
import threading
from oslo_utils import reflection
from oslo_utils import strutils
import requests
from novaclient import exceptions
from novaclient import utils
class BytesWithMeta(bytes, RequestIdMixin):

    def __new__(cls, value, resp):
        return super(BytesWithMeta, cls).__new__(cls, value)

    def __init__(self, values, resp):
        self.request_ids_setup()
        self.append_request_ids(resp)