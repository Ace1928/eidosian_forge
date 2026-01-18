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
def _id_to_dn(self, object_id):
    if self.LDAP_SCOPE == ldap.SCOPE_ONELEVEL:
        return self._id_to_dn_string(object_id)
    with self.get_connection() as conn:
        search_result = conn.search_s(self.tree_dn, self.LDAP_SCOPE, u'(&(%(id_attr)s=%(id)s)(objectclass=%(objclass)s))' % {'id_attr': self.id_attr, 'id': ldap.filter.escape_filter_chars(str(object_id)), 'objclass': self.object_class}, attrlist=DN_ONLY)
    if search_result:
        dn, attrs = search_result[0]
        return dn
    else:
        return self._id_to_dn_string(object_id)