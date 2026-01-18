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
@classmethod
def _alternate_id(cls):
    """Return the name of any value known as an alternate_id

        NOTE: This will only ever return the first such alternate_id.
        Only one alternate_id should be specified.

        Returns an empty string if no name exists, as this method is
        consumed by _get_id and passed to getattr.
        """
    for value in cls.__dict__.values():
        if isinstance(value, Body):
            if value.alternate_id:
                return value.name
    return ''