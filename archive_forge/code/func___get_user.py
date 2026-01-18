import json
import os
from collections import defaultdict
import hashlib
import tempfile
from functools import partial
import kubernetes.dynamic
import kubernetes.dynamic.discovery
from kubernetes import __version__
from kubernetes.dynamic.exceptions import (
from ansible_collections.kubernetes.core.plugins.module_utils.client.resource import (
def __get_user(self):
    if hasattr(os, 'getlogin'):
        try:
            user = os.getlogin()
            if user:
                return str(user)
        except OSError:
            pass
    if hasattr(os, 'getuid'):
        try:
            user = os.getuid()
            if user:
                return str(user)
        except OSError:
            pass
    user = os.environ.get('USERNAME')
    if user:
        return str(user)
    return None