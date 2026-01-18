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
def _determine_calling_package():
    """Walk the call frames trying to identify what is using this module."""
    mod_lookup = dict(((m.__file__, n) for n, m in sys.modules.copy().items() if hasattr(m, '__file__')))
    ignored = ('debtcollector', 'keystoneauth1', 'keystoneclient')
    i = 0
    while True:
        i += 1
        try:
            f = sys._getframe(i)
            try:
                name = mod_lookup[f.f_code.co_filename]
                name, _, _ = name.partition('.')
                if name not in ignored:
                    return name
            except KeyError:
                pass
        except ValueError:
            break
    return ''