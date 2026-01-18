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
def _validate_networks(self, networks, key):
    value = None
    for net in networks:
        if value and net[key]:
            raise exceptions.ConfigException('Duplicate network entries for {key}: {net1} and {net2}. Only one network can be flagged with {key}'.format(key=key, net1=value['name'], net2=net['name']))
        if not value and net[key]:
            value = net