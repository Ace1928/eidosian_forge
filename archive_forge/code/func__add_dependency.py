import collections
import copy
import datetime
import hashlib
import inspect
from unittest import mock
import iso8601
from oslo_versionedobjects import base
from oslo_versionedobjects import exception
from oslo_versionedobjects import fields
from oslo_versionedobjects import fixture
from oslo_versionedobjects import test
def _add_dependency(self, parent_cls, child_cls, tree):
    deps = tree.get(parent_cls.__name__, {})
    deps[child_cls.__name__] = '1.0'
    tree[parent_cls.__name__] = deps