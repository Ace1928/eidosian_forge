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
def _dictify_resource(resource):
    if isinstance(resource, list):
        return [_dictify_resource(r) for r in resource]
    elif hasattr(resource, 'toDict'):
        return resource.toDict()
    else:
        return resource