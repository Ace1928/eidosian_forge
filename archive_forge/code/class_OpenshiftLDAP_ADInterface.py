from __future__ import (absolute_import, division, print_function)
import os
import copy
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.six import iteritems
class OpenshiftLDAP_ADInterface(object):

    def __init__(self, connection, user_query, group_member_attr, user_name_attr):
        self.connection = connection
        self.userQuery = user_query
        self.groupMembershipAttributes = group_member_attr
        self.userNameAttributes = user_name_attr
        self.required_user_attributes = self.userNameAttributes or []
        for attr in self.groupMembershipAttributes:
            if attr not in self.required_user_attributes:
                self.required_user_attributes.append(attr)
        self.cache = {}
        self.cache_populated = False

    def is_entry_present(self, cache_item, entry):
        for item in cache_item:
            if item[0] == entry[0]:
                return True
        return False

    def populate_cache(self):
        if not self.cache_populated:
            self.cache_populated = True
            entries, err = self.userQuery.ldap_search(self.connection, self.required_user_attributes)
            if err:
                return err
            for entry in entries:
                for group_attr in self.groupMembershipAttributes:
                    uids = openshift_ldap_get_attribute_for_entry(entry, group_attr)
                    if not isinstance(uids, list):
                        uids = [uids]
                    for uid in uids:
                        if uid not in self.cache:
                            self.cache[uid] = []
                        if not self.is_entry_present(self.cache[uid], entry):
                            self.cache[uid].append(entry)
        return None

    def list_groups(self):
        err = self.populate_cache()
        if err:
            return (None, err)
        result = []
        if self.cache:
            result = self.cache.keys()
        return (result, None)

    def extract_members(self, uid):
        if uid in self.cache:
            return (self.cache[uid], None)
        users_in_group = []
        for attr in self.groupMembershipAttributes:
            query_on_attribute = OpenshiftLDAPQueryOnAttribute(self.userQuery.qry, attr)
            entries, error = query_on_attribute.ldap_search(self.connection, uid, self.required_user_attributes, unique_entry=False)
            if error and 'not found' not in error:
                return (None, error)
            if not entries:
                continue
            for entry in entries:
                if not self.is_entry_present(users_in_group, entry):
                    users_in_group.append(entry)
        self.cache[uid] = users_in_group
        return (users_in_group, None)