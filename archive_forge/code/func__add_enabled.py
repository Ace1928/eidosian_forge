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
def _add_enabled(self, object_id):
    member_attr_val = self._id_to_member_attribute_value(object_id)
    with self.get_connection() as conn:
        if not self._is_member_enabled(member_attr_val, conn):
            modlist = [(ldap.MOD_ADD, self.member_attribute, [member_attr_val])]
            try:
                conn.modify_s(self.enabled_emulation_dn, modlist)
            except ldap.NO_SUCH_OBJECT:
                attr_list = [('objectClass', [self.group_objectclass]), (self.member_attribute, [member_attr_val]), self.enabled_emulation_naming_attr]
                conn.add_s(self.enabled_emulation_dn, attr_list)