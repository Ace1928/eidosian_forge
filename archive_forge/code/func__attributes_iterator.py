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
def _attributes_iterator(cls, components=tuple([Body, Header])):
    """Iterator over all Resource attributes"""
    for klass in cls.__mro__:
        for attr, component in klass.__dict__.items():
            if isinstance(component, components):
                yield (attr, component)