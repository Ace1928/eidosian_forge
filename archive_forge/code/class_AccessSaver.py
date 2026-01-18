from collections.abc import Mapping
import hashlib
import queue
import string
import threading
import time
import typing as ty
import keystoneauth1
from keystoneauth1 import adapter as ks_adapter
from keystoneauth1 import discover
from openstack import _log
from openstack import exceptions
class AccessSaver:

    def __init__(self):
        self.keys = []

    def __getitem__(self, key):
        self.keys.append(key)