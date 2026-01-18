from __future__ import absolute_import, division, print_function
import json
import logging
import optparse
import os
import ssl
import sys
import time
from ansible.module_utils.common._collections_compat import MutableMapping
from ansible.module_utils.six import integer_types, text_type, string_types
from ansible.module_utils.six.moves import configparser
from psphere.client import Client
from psphere.errors import ObjectNotFoundError
from psphere.managedobjects import HostSystem, VirtualMachine, ManagedObject, ClusterComputeResource
from suds.sudsobject import Object as SudsObject
def _get_obj_info(self, obj, depth=99, seen=None):
    """
        Recursively build a data structure for the given pSphere object (depth
        only applies to ManagedObject instances).
        """
    seen = seen or set()
    if isinstance(obj, ManagedObject):
        try:
            obj_unicode = text_type(getattr(obj, 'name'))
        except AttributeError:
            obj_unicode = ()
        if obj in seen:
            return obj_unicode
        seen.add(obj)
        if depth <= 0:
            return obj_unicode
        d = {}
        for attr in dir(obj):
            if attr.startswith('_'):
                continue
            try:
                val = getattr(obj, attr)
                obj_info = self._get_obj_info(val, depth - 1, seen)
                if obj_info != ():
                    d[attr] = obj_info
            except Exception as e:
                pass
        return d
    elif isinstance(obj, SudsObject):
        d = {}
        for key, val in iter(obj):
            obj_info = self._get_obj_info(val, depth, seen)
            if obj_info != ():
                d[key] = obj_info
        return d
    elif isinstance(obj, (list, tuple)):
        l = []
        for val in iter(obj):
            obj_info = self._get_obj_info(val, depth, seen)
            if obj_info != ():
                l.append(obj_info)
        return l
    elif isinstance(obj, (type(None), bool, float) + string_types + integer_types):
        return obj
    else:
        return ()