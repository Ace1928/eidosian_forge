from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.elementsw.plugins.module_utils.netapp as netapp_utils
class NetappElementLdap(object):

    def __init__(self):
        self.argument_spec = netapp_utils.ontap_sf_host_argument_spec()
        self.argument_spec.update(state=dict(required=False, type='str', choices=['present', 'absent'], default='present'), authType=dict(type='str', choices=['DirectBind', 'SearchAndBind']), groupSearchBaseDn=dict(type='str'), groupSearchType=dict(type='str', choices=['NoGroup', 'ActiveDirectory', 'MemberDN']), serverURIs=dict(type='str'), userSearchBaseDN=dict(type='str'), searchBindDN=dict(type='str'), searchBindPassword=dict(type='str', no_log=True), userSearchFilter=dict(type='str'), userDNTemplate=dict(type='str'), groupSearchCustomFilter=dict(type='str'))
        self.module = AnsibleModule(argument_spec=self.argument_spec, supports_check_mode=True)
        param = self.module.params
        self.state = param['state']
        self.authType = param['authType']
        self.groupSearchBaseDn = param['groupSearchBaseDn']
        self.groupSearchType = param['groupSearchType']
        self.serverURIs = param['serverURIs']
        if self.serverURIs is not None:
            self.serverURIs = self.serverURIs.split(',')
        self.userSearchBaseDN = param['userSearchBaseDN']
        self.searchBindDN = param['searchBindDN']
        self.searchBindPassword = param['searchBindPassword']
        self.userSearchFilter = param['userSearchFilter']
        self.userDNTemplate = param['userDNTemplate']
        self.groupSearchCustomFilter = param['groupSearchCustomFilter']
        if HAS_SF_SDK is False:
            self.module.fail_json(msg='Unable to import the SolidFire Python SDK')
        else:
            self.sfe = netapp_utils.create_sf_connection(module=self.module)

    def get_ldap_configuration(self):
        """
            Return ldap configuration if found

            :return: Details about the ldap configuration. None if not found.
            :rtype: solidfire.models.GetLdapConfigurationResult
        """
        ldap_config = self.sfe.get_ldap_configuration()
        return ldap_config

    def enable_ldap(self):
        """
        Enable LDAP
        :return: nothing
        """
        try:
            self.sfe.enable_ldap_authentication(self.serverURIs, auth_type=self.authType, group_search_base_dn=self.groupSearchBaseDn, group_search_type=self.groupSearchType, group_search_custom_filter=self.groupSearchCustomFilter, search_bind_dn=self.searchBindDN, search_bind_password=self.searchBindPassword, user_search_base_dn=self.userSearchBaseDN, user_search_filter=self.userSearchFilter, user_dntemplate=self.userDNTemplate)
        except solidfire.common.ApiServerError as error:
            self.module.fail_json(msg='Error enabling LDAP: %s' % to_native(error), exception=traceback.format_exc())

    def check_config(self, ldap_config):
        """
        Check to see if the ldap config has been modified.
        :param ldap_config: The LDAP configuration
        :return: False if the config is the same as the playbook, True if it is not
        """
        if self.authType != ldap_config.ldap_configuration.auth_type:
            return True
        if self.serverURIs != ldap_config.ldap_configuration.server_uris:
            return True
        if self.groupSearchBaseDn != ldap_config.ldap_configuration.group_search_base_dn:
            return True
        if self.groupSearchType != ldap_config.ldap_configuration.group_search_type:
            return True
        if self.groupSearchCustomFilter != ldap_config.ldap_configuration.group_search_custom_filter:
            return True
        if self.searchBindDN != ldap_config.ldap_configuration.search_bind_dn:
            return True
        if self.searchBindPassword != ldap_config.ldap_configuration.search_bind_password:
            return True
        if self.userSearchBaseDN != ldap_config.ldap_configuration.user_search_base_dn:
            return True
        if self.userSearchFilter != ldap_config.ldap_configuration.user_search_filter:
            return True
        if self.userDNTemplate != ldap_config.ldap_configuration.user_dntemplate:
            return True
        return False

    def apply(self):
        changed = False
        ldap_config = self.get_ldap_configuration()
        if self.state == 'absent':
            if ldap_config and ldap_config.ldap_configuration.enabled:
                changed = True
        if self.state == 'present' and self.check_config(ldap_config):
            changed = True
        if changed:
            if self.module.check_mode:
                pass
            elif self.state == 'present':
                self.enable_ldap()
            elif self.state == 'absent':
                self.sfe.disable_ldap_authentication()
        self.module.exit_json(changed=changed)