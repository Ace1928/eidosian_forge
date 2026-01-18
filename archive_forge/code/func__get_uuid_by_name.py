from __future__ import absolute_import, division, print_function
import os
import sys
import copy
import json
import logging
import time
from datetime import datetime, timedelta
from ssl import SSLError
def _get_uuid_by_name(self, path, name, tenant='admin', tenant_uuid='', api_version=None):
    """gets object by name and service path and returns uuid"""
    resp = self.get_object_by_name(path, name, tenant, tenant_uuid, api_version=api_version)
    if not resp:
        raise ObjectNotFound('%s/%s' % (path, name))
    return self.get_obj_uuid(resp)