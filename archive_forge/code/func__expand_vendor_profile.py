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
def _expand_vendor_profile(self, name, cloud, our_cloud):
    profile_name = our_cloud.get('profile', our_cloud.get('cloud', None))
    if not profile_name or profile_name == self.envvar_key:
        return
    if 'cloud' in our_cloud:
        warnings.warn(f"{self.config_filename} uses the keyword 'cloud' to reference a known vendor profile. This has been deprecated in favor of the 'profile' keyword.", os_warnings.OpenStackDeprecationWarning)
    vendor_filename, vendor_file = self._load_vendor_file()
    if vendor_file and 'public-clouds' in vendor_file and (profile_name in vendor_file['public-clouds']):
        _auth_update(cloud, vendor_file['public-clouds'][profile_name])
    else:
        profile_data = vendors.get_profile(profile_name)
        if profile_data:
            nested_profile = profile_data.pop('profile', None)
            if nested_profile:
                nested_profile_data = vendors.get_profile(nested_profile)
                if nested_profile_data:
                    profile_data = nested_profile_data
            status = profile_data.pop('status', 'active')
            message = profile_data.pop('message', '')
            if status == 'deprecated':
                warnings.warn(f'{profile_name} is deprecated: {message}', os_warnings.OpenStackDeprecationWarning)
            elif status == 'shutdown':
                raise exceptions.ConfigException('{profile_name} references a cloud that no longer exists: {message}'.format(profile_name=profile_name, message=message))
            _auth_update(cloud, profile_data)
        else:
            warnings.warn(f"Couldn't find the vendor profile {profile_name} for the cloud {name}", os_warnings.ConfigurationWarning)