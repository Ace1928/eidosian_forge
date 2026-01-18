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
def _construct_session(session_obj=None):
    if not session_obj:
        session_obj = requests.Session()
        for scheme in list(session_obj.adapters):
            session_obj.mount(scheme, TCPKeepAliveAdapter())
    return session_obj