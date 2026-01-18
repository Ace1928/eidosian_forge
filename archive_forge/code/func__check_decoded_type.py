import collections.abc
import contextlib
import datetime
import functools
import inspect
import io
import os
import re
import socket
import sys
import threading
import types
import enum
from oslo_serialization import jsonutils
from oslo_serialization import msgpackutils
from oslo_utils import encodeutils
from oslo_utils import importutils
from oslo_utils import netutils
from oslo_utils import reflection
from taskflow.types import failure
def _check_decoded_type(data, root_types=(dict,)):
    if root_types:
        if not isinstance(root_types, tuple):
            root_types = tuple(root_types)
        if not isinstance(data, root_types):
            if len(root_types) == 1:
                root_type = root_types[0]
                raise ValueError("Expected '%s' root type not '%s'" % (root_type, type(data)))
            else:
                raise ValueError("Expected %s root types not '%s'" % (list(root_types), type(data)))
    return data