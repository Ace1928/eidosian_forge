import abc
import collections
import inspect
import itertools
import operator
import typing as ty
import urllib.parse
import warnings
import jsonpatch
from keystoneauth1 import adapter
from keystoneauth1 import discover
from requests import structures
from openstack import _log
from openstack import exceptions
from openstack import format
from openstack import utils
from openstack import warnings as os_warnings
def _to_munch(self, original_names=True):
    """Convert this resource into a Munch compatible with shade."""
    return self.to_dict(body=True, headers=False, original_names=original_names, _to_munch=True)