from __future__ import (absolute_import, division, print_function)
import os
import copy
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.six import iteritems
class OpenshiftLDAP_AugmentedADInterface(OpenshiftLDAP_ADInterface):

    def __init__(self, connection, user_query, group_member_attr, user_name_attr, group_qry, group_name_attr):
        super(OpenshiftLDAP_AugmentedADInterface, self).__init__(connection, user_query, group_member_attr, user_name_attr)
        self.groupQuery = copy.deepcopy(group_qry)
        self.groupNameAttributes = group_name_attr
        self.required_group_attributes = [self.groupQuery.query_attribute]
        for x in self.groupNameAttributes:
            if x not in self.required_group_attributes:
                self.required_group_attributes.append(x)
        self.cached_groups = {}

    def get_group_entry(self, uid):
        """
            get_group_entry returns an LDAP group entry for the given group UID by searching the internal cache
            of the LDAPInterface first, then sending an LDAP query if the cache did not contain the entry.
        """
        if uid in self.cached_groups:
            return (self.cached_groups.get(uid), None)
        group, err = self.groupQuery.ldap_search(self.connection, uid, self.required_group_attributes)
        if err:
            return (None, err)
        self.cached_groups[uid] = group
        return (group, None)

    def exists(self, ldapuid):
        members, error = self.extract_members(ldapuid)
        if error:
            return (False, error)
        group_exists = bool(members)
        entry, error = self.get_group_entry(ldapuid)
        if error:
            if 'not found' in error:
                return (False, None)
            else:
                return (False, error)
        else:
            return (group_exists and bool(entry), None)