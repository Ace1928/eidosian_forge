import argparse as argparse_mod
import collections
import copy
import errno
import json
import os
import re
import sys
import typing as ty
import warnings
from keystoneauth1 import adapter
from keystoneauth1 import loading
import platformdirs
import yaml
from openstack import _log
from openstack.config import _util
from openstack.config import cloud_region
from openstack.config import defaults
from openstack.config import vendors
from openstack import exceptions
from openstack import warnings as os_warnings
def _fix_backwards_api_timeout(self, cloud):
    new_cloud = {}
    service_timeout = None
    for key in cloud.keys():
        if key.endswith('timeout') and (not (key == 'timeout' or key == 'api_timeout')):
            service_timeout = cloud[key]
        else:
            new_cloud[key] = cloud[key]
    if service_timeout is not None:
        new_cloud['api_timeout'] = service_timeout
    if self._argv_timeout:
        if 'timeout' in new_cloud and new_cloud['timeout']:
            new_cloud['api_timeout'] = new_cloud.pop('timeout')
    return new_cloud