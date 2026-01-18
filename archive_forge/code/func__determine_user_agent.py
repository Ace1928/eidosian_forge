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
def _determine_user_agent():
    """Attempt to programmatically generate a user agent string.

    First, look at the name of the process. Return this unless it is in
    the `ignored` list.  Otherwise, look at the function call stack and
    try to find the name of the code that invoked this module.
    """
    ignored = ('mod_wsgi',)
    try:
        name = sys.argv[0]
    except IndexError:
        return ''
    if not name:
        return ''
    name = os.path.basename(name)
    if name in ignored:
        name = _determine_calling_package()
    return name