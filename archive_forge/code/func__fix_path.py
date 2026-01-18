import importlib
import logging
import os
import sys
import warnings
from keystoneauth1 import adapter
from keystoneauth1 import session as ks_session
from oslo_utils import importutils
from barbicanclient import exceptions
def _fix_path(self, path):
    if not path[-1] == '/':
        path += '/'
    return path