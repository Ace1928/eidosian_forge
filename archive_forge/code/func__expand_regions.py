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
def _expand_regions(self, regions):
    ret = []
    for region in regions:
        if isinstance(region, dict):
            if 'name' not in region or not {'name', 'values'} >= set(region):
                raise exceptions.ConfigException('Invalid region entry at: %s' % region)
            if 'values' not in region:
                region['values'] = {}
            ret.append(copy.deepcopy(region))
        else:
            ret.append(self._expand_region_name(region))
    return ret