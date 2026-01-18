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
def _get_base_cloud_config(self, name, profile=None):
    cloud = dict()
    if name and name not in self.cloud_config['clouds']:
        raise exceptions.ConfigException('Cloud {name} was not found.'.format(name=name))
    our_cloud = self.cloud_config['clouds'].get(name, dict())
    if profile:
        our_cloud['profile'] = profile
    cloud.update(self.defaults)
    self._expand_vendor_profile(name, cloud, our_cloud)
    if 'auth' not in cloud:
        cloud['auth'] = dict()
    _auth_update(cloud, our_cloud)
    if 'cloud' in cloud:
        del cloud['cloud']
    return cloud