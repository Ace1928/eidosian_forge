from __future__ import (absolute_import, division, print_function)
import os
import copy
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.six import iteritems
class OpenshiftLDAPInterface(object):

    def __init__(self, connection, groupQuery, groupNameAttributes, groupMembershipAttributes, userQuery, userNameAttributes, config):
        self.connection = connection
        self.groupQuery = copy.deepcopy(groupQuery)
        self.groupNameAttributes = groupNameAttributes
        self.groupMembershipAttributes = groupMembershipAttributes
        self.userQuery = copy.deepcopy(userQuery)
        self.userNameAttributes = userNameAttributes
        self.config = config
        self.tolerate_not_found = boolean(config.get('tolerateMemberNotFoundErrors', False))
        self.tolerate_out_of_scope = boolean(config.get('tolerateMemberOutOfScopeErrors', False))
        self.required_group_attributes = [self.groupQuery.query_attribute]
        for x in self.groupNameAttributes + self.groupMembershipAttributes:
            if x not in self.required_group_attributes:
                self.required_group_attributes.append(x)
        self.required_user_attributes = [self.userQuery.query_attribute]
        for x in self.userNameAttributes:
            if x not in self.required_user_attributes:
                self.required_user_attributes.append(x)
        self.cached_groups = {}
        self.cached_users = {}

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

    def get_user_entry(self, uid):
        """
            get_user_entry returns an LDAP group entry for the given user UID by searching the internal cache
            of the LDAPInterface first, then sending an LDAP query if the cache did not contain the entry.
        """
        if uid in self.cached_users:
            return (self.cached_users.get(uid), None)
        entry, err = self.userQuery.ldap_search(self.connection, uid, self.required_user_attributes)
        if err:
            return (None, err)
        self.cached_users[uid] = entry
        return (entry, None)

    def exists(self, ldapuid):
        group, error = self.get_group_entry(ldapuid)
        return (bool(group), error)

    def list_groups(self):
        group_qry = copy.deepcopy(self.groupQuery.qry)
        group_qry['attrlist'] = self.required_group_attributes
        groups, err = openshift_ldap_query_for_entries(connection=self.connection, qry=group_qry, unique_entry=False)
        if err:
            return (None, err)
        group_uids = []
        for entry in groups:
            uid = openshift_ldap_get_attribute_for_entry(entry, self.groupQuery.query_attribute)
            if not uid:
                return (None, 'Unable to find LDAP group uid for entry %s' % entry)
            self.cached_groups[uid] = entry
            group_uids.append(uid)
        return (group_uids, None)

    def extract_members(self, uid):
        """
            returns the LDAP member entries for a group specified with a ldapGroupUID
        """
        group, err = self.get_group_entry(uid)
        if err:
            return (None, err)
        member_uids = []
        for attribute in self.groupMembershipAttributes:
            member_uids += openshift_ldap_get_attribute_for_entry(group, attribute)
        members = []
        for user_uid in member_uids:
            entry, err = self.get_user_entry(user_uid)
            if err:
                if self.tolerate_not_found and err.startswith('Entry not found'):
                    continue
                elif err == LDAP_SEARCH_OUT_OF_SCOPE_ERROR:
                    continue
                return (None, err)
            members.append(entry)
        return (members, None)