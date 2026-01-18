import os
from base64 import b64encode
from collections import deque
from http.client import HTTPConnection
from json import loads
from threading import Event, Thread
from time import sleep
from urllib.parse import urlparse, urlunparse
import requests
from kivy.clock import Clock
from kivy.config import Config
from kivy.logger import Logger
from kivy.utils import platform
from kivy.weakmethod import WeakMethod
def _get_connection_for_scheme(self, scheme):
    """Return the Connection class for a particular scheme.
        This is an internal function that can be expanded to support custom
        schemes.

        Actual supported schemes: http, https.
        """
    if scheme == 'http':
        return HTTPConnection
    elif scheme == 'https' and HTTPSConnection is not None:
        return HTTPSConnection
    else:
        raise Exception('No class for scheme %s' % scheme)