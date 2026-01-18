import glob
import json
import os
import typing as ty
import urllib
import requests
import yaml
from openstack.config import _util
from openstack import exceptions
def _get_vendor_defaults():
    global _VENDOR_DEFAULTS
    if not _VENDOR_DEFAULTS:
        for vendor in glob.glob(os.path.join(_VENDORS_PATH, '*.yaml')):
            with open(vendor, 'r') as f:
                vendor_data = yaml.safe_load(f)
                _VENDOR_DEFAULTS[vendor_data['name']] = vendor_data['profile']
        for vendor in glob.glob(os.path.join(_VENDORS_PATH, '*.json')):
            with open(vendor, 'r') as f:
                vendor_data = json.load(f)
                _VENDOR_DEFAULTS[vendor_data['name']] = vendor_data['profile']
    return _VENDOR_DEFAULTS