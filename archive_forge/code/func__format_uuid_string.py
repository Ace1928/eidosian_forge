import contextlib
import fnmatch
import inspect
import re
import uuid
from decorator import decorator
import jmespath
import netifaces
from openstack import _log
from openstack import exceptions
def _format_uuid_string(string):
    return string.replace('urn:', '').replace('uuid:', '').strip('{}').replace('-', '').lower()