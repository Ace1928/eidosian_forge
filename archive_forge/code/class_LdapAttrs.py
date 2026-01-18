from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native, to_bytes, to_text
from ansible_collections.community.general.plugins.module_utils.ldap import LdapGeneric, gen_specs, ldap_required_together
import re
class LdapAttrs(LdapGeneric):

    def __init__(self, module):
        LdapGeneric.__init__(self, module)
        self.attrs = self.module.params['attributes']
        self.state = self.module.params['state']
        self.ordered = self.module.params['ordered']

    def _order_values(self, values):
        """ Prepend X-ORDERED index numbers to attribute's values. """
        ordered_values = []
        if isinstance(values, list):
            for index, value in enumerate(values):
                cleaned_value = re.sub('^\\{\\d+\\}', '', value)
                ordered_values.append('{' + str(index) + '}' + cleaned_value)
        return ordered_values

    def _normalize_values(self, values):
        """ Normalize attribute's values. """
        norm_values = []
        if isinstance(values, list):
            if self.ordered:
                norm_values = list(map(to_bytes, self._order_values(list(map(str, values)))))
            else:
                norm_values = list(map(to_bytes, values))
        else:
            norm_values = [to_bytes(str(values))]
        return norm_values

    def add(self):
        modlist = []
        new_attrs = {}
        for name, values in self.module.params['attributes'].items():
            norm_values = self._normalize_values(values)
            added_values = []
            for value in norm_values:
                if self._is_value_absent(name, value):
                    modlist.append((ldap.MOD_ADD, name, value))
                    added_values.append(value)
            if added_values:
                new_attrs[name] = norm_values
        return (modlist, {}, new_attrs)

    def delete(self):
        modlist = []
        old_attrs = {}
        new_attrs = {}
        for name, values in self.module.params['attributes'].items():
            norm_values = self._normalize_values(values)
            removed_values = []
            for value in norm_values:
                if self._is_value_present(name, value):
                    removed_values.append(value)
                    modlist.append((ldap.MOD_DELETE, name, value))
            if removed_values:
                old_attrs[name] = norm_values
                new_attrs[name] = [value for value in norm_values if value not in removed_values]
        return (modlist, old_attrs, new_attrs)

    def exact(self):
        modlist = []
        old_attrs = {}
        new_attrs = {}
        for name, values in self.module.params['attributes'].items():
            norm_values = self._normalize_values(values)
            try:
                results = self.connection.search_s(self.dn, ldap.SCOPE_BASE, attrlist=[name])
            except ldap.LDAPError as e:
                self.fail('Cannot search for attribute %s' % name, e)
            current = results[0][1].get(name, [])
            if frozenset(norm_values) != frozenset(current):
                if len(current) == 0:
                    modlist.append((ldap.MOD_ADD, name, norm_values))
                elif len(norm_values) == 0:
                    modlist.append((ldap.MOD_DELETE, name, None))
                else:
                    modlist.append((ldap.MOD_REPLACE, name, norm_values))
                old_attrs[name] = current
                new_attrs[name] = norm_values
                if len(current) == 1 and len(norm_values) == 1:
                    old_attrs[name] = current[0]
                    new_attrs[name] = norm_values[0]
        return (modlist, old_attrs, new_attrs)

    def _is_value_present(self, name, value):
        """ True if the target attribute has the given value. """
        try:
            escaped_value = ldap.filter.escape_filter_chars(to_text(value))
            filterstr = '(%s=%s)' % (name, escaped_value)
            dns = self.connection.search_s(self.dn, ldap.SCOPE_BASE, filterstr)
            is_present = len(dns) == 1
        except ldap.NO_SUCH_OBJECT:
            is_present = False
        return is_present

    def _is_value_absent(self, name, value):
        """ True if the target attribute doesn't have the given value. """
        return not self._is_value_present(name, value)