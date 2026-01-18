import collections
import collections.abc
import datetime
import functools
import io
import ipaddress
import itertools
import json
from unittest import mock
from xmlrpc import client as xmlrpclib
import netaddr
from oslo_i18n import fixture
from oslotest import base as test_base
from oslo_serialization import jsonutils
class MappingClass(collections.abc.Mapping):

    def __init__(self):
        self.data = dict(a=1, b=2, c=3)

    def __getitem__(self, val):
        return self.data[val]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)