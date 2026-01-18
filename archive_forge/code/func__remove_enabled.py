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
def _remove_enabled(self, object_id):
    member_attr_val = self._id_to_member_attribute_value(object_id)
    modlist = [(ldap.MOD_DELETE, self.member_attribute, [member_attr_val])]
    with self.get_connection() as conn:
        try:
            conn.modify_s(self.enabled_emulation_dn, modlist)
        except (ldap.NO_SUCH_OBJECT, ldap.NO_SUCH_ATTRIBUTE):
            pass