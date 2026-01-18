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
def _ldap_get_limited(self, base, scope, filterstr, attrlist, sizelimit):
    with self.get_connection() as conn:
        try:
            control = ldap.controls.libldap.SimplePagedResultsControl(criticality=True, size=sizelimit, cookie='')
            msgid = conn.search_ext(base, scope, filterstr, attrlist, serverctrls=[control])
            rdata = conn.result3(msgid)
            return rdata
        except ldap.NO_SUCH_OBJECT:
            return []