import abc
import codecs
import os.path
import random
import re
import sys
import uuid
import weakref
import ldap.controls
import ldap.filter
import ldappool
from oslo_log import log
from oslo_utils import reflection
from keystone.common import driver_hints
from keystone import exception
from keystone.i18n import _
@staticmethod
def _parse_extra_attrs(option_list):
    mapping = {}
    for item in option_list:
        try:
            ldap_attr, attr_map = item.split(':')
        except ValueError:
            LOG.warning('Invalid additional attribute mapping: "%s". Format must be <ldap_attribute>:<keystone_attribute>', item)
            continue
        mapping[ldap_attr] = attr_map
    return mapping