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
def _consume_mapped_attrs(self, mapping_cls, attrs):
    mapping = self._get_mapping(mapping_cls)
    return self._consume_attrs(mapping, attrs)