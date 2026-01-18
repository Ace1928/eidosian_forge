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
def _is_member_enabled(self, member_attr_val, conn):
    query = '(%s=%s)' % (self.member_attribute, ldap.filter.escape_filter_chars(member_attr_val))
    try:
        enabled_value = conn.search_s(self.enabled_emulation_dn, ldap.SCOPE_BASE, query, attrlist=DN_ONLY)
    except ldap.NO_SUCH_OBJECT:
        return False
    else:
        return bool(enabled_value)