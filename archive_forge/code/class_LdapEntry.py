from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native, to_bytes
from ansible_collections.community.general.plugins.module_utils.ldap import LdapGeneric, gen_specs, ldap_required_together
class LdapEntry(LdapGeneric):

    def __init__(self, module):
        LdapGeneric.__init__(self, module)
        self.state = self.module.params['state']
        self.recursive = self.module.params['recursive']
        self.module.params['attributes']['objectClass'] = self.module.params['objectClass']
        if self.state == 'present':
            self.attrs = self._load_attrs()

    def _load_attrs(self):
        """ Turn attribute's value to array. """
        attrs = {}
        for name, value in self.module.params['attributes'].items():
            if isinstance(value, list):
                attrs[name] = list(map(to_bytes, value))
            else:
                attrs[name] = [to_bytes(value)]
        return attrs

    def add(self):
        """ If self.dn does not exist, returns a callable that will add it. """

        def _add():
            self.connection.add_s(self.dn, modlist)
        if not self._is_entry_present():
            modlist = ldap.modlist.addModlist(self.attrs)
            action = _add
        else:
            action = None
        return action

    def delete(self):
        """ If self.dn exists, returns a callable that will delete either
        the item itself if the recursive option is not set or the whole branch
        if it is. """

        def _delete():
            self.connection.delete_s(self.dn)

        def _delete_recursive():
            """ Attempt recursive deletion using the subtree-delete control.
            If that fails, do it manually. """
            try:
                subtree_delete = ldap.controls.ValueLessRequestControl('1.2.840.113556.1.4.805')
                self.connection.delete_ext_s(self.dn, serverctrls=[subtree_delete])
            except ldap.NOT_ALLOWED_ON_NONLEAF:
                search = self.connection.search_s(self.dn, ldap.SCOPE_SUBTREE, attrlist=('dn',))
                search.reverse()
                for entry in search:
                    self.connection.delete_s(entry[0])
        if self._is_entry_present():
            if self.recursive:
                action = _delete_recursive
            else:
                action = _delete
        else:
            action = None
        return action

    def _is_entry_present(self):
        try:
            self.connection.search_s(self.dn, ldap.SCOPE_BASE)
        except ldap.NO_SUCH_OBJECT:
            is_present = False
        else:
            is_present = True
        return is_present