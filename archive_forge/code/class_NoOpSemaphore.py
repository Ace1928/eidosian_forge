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
class NoOpSemaphore(object):
    """Empty context manager for use as a default semaphore."""

    def __enter__(self):
        """Enter the context manager and do nothing."""
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the context manager and do nothing."""
        pass