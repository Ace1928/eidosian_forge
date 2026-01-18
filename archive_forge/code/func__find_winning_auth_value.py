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
def _find_winning_auth_value(self, opt, config):
    opt_name = opt.name.replace('-', '_')
    if opt_name in config:
        return config[opt_name]
    else:
        deprecated = getattr(opt, 'deprecated', getattr(opt, 'deprecated_opts', []))
        for d_opt in deprecated:
            d_opt_name = d_opt.name.replace('-', '_')
            if d_opt_name in config:
                return config[d_opt_name]