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
def _handle_domain_id(self, cloud):
    mappings = {'domain_id': ('user_domain_id', 'project_domain_id'), 'domain_name': ('user_domain_name', 'project_domain_name')}
    for target_key, possible_values in mappings.items():
        if not self._project_scoped(cloud):
            if target_key in cloud and target_key not in cloud['auth']:
                cloud['auth'][target_key] = cloud.pop(target_key)
            continue
        for key in possible_values:
            if target_key in cloud['auth'] and key not in cloud['auth']:
                cloud['auth'][key] = cloud['auth'][target_key]
        cloud.pop(target_key, None)
        cloud['auth'].pop(target_key, None)
    return cloud