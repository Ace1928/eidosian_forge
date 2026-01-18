import glob
import hashlib
import importlib.util
import itertools
import json
import logging
import os
import pkgutil
import re
import urllib
from urllib import parse as urlparse
from keystoneauth1 import access
from keystoneauth1 import adapter
from keystoneauth1 import discover
from keystoneauth1.identity import base
from oslo_utils import encodeutils
from oslo_utils import importutils
from oslo_utils import strutils
import requests
from cinderclient._i18n import _
from cinderclient import api_versions
from cinderclient import exceptions
import cinderclient.extension
def _discover_via_python_path():
    for module_loader, name, ispkg in pkgutil.iter_modules():
        if name.endswith('cinderclient_ext'):
            if not hasattr(module_loader, 'load_module'):
                module_loader = module_loader.find_module(name)
            module = module_loader.load_module(name)
            yield (name, module)