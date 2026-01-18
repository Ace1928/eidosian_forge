from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
class AaaServer(object):
    """ Manages aaa configuration """

    def netconf_get_config(self, **kwargs):
        """ Get configure by netconf """
        module = kwargs['module']
        conf_str = kwargs['conf_str']
        xml_str = get_nc_config(module, conf_str)
        return xml_str

    def netconf_set_config(self, **kwargs):
        """ Set configure by netconf """
        module = kwargs['module']
        conf_str = kwargs['conf_str']
        recv_xml = set_nc_config(module, conf_str)
        return recv_xml

    def get_authentication_scheme(self, **kwargs):
        """ Get scheme of authentication """
        module = kwargs['module']
        conf_str = CE_GET_AUTHENTICATION_SCHEME
        xml_str = self.netconf_get_config(module=module, conf_str=conf_str)
        result = list()
        if '<data/>' in xml_str:
            return result
        else:
            re_find = re.findall('.*<authenSchemeName>(.*)</authenSchemeName>.*\\s*<firstAuthenMode>(.*)</firstAuthenMode>.*\\s*<secondAuthenMode>(.*)</secondAuthenMode>.*\\s*', xml_str)
            if re_find:
                return re_find
            else:
                return result

    def get_authentication_domain(self, **kwargs):
        """ Get domain of authentication """
        module = kwargs['module']
        conf_str = CE_GET_AUTHENTICATION_DOMAIN
        xml_str = self.netconf_get_config(module=module, conf_str=conf_str)
        result = list()
        if '<data/>' in xml_str:
            return result
        else:
            re_find = re.findall('.*<domainName>(.*)</domainName>.*\\s*<authenSchemeName>(.*)</authenSchemeName>.*', xml_str)
            if re_find:
                return re_find
            else:
                return result

    def merge_authentication_scheme(self, **kwargs):
        """ Merge scheme of authentication """
        authen_scheme_name = kwargs['authen_scheme_name']
        first_authen_mode = kwargs['first_authen_mode']
        module = kwargs['module']
        conf_str = CE_MERGE_AUTHENTICATION_SCHEME % (authen_scheme_name, first_authen_mode)
        xml = self.netconf_set_config(module=module, conf_str=conf_str)
        if '<ok/>' not in xml:
            module.fail_json(msg='Error: Merge authentication scheme failed.')
        cmds = []
        cmd = 'authentication-scheme %s' % authen_scheme_name
        cmds.append(cmd)
        cmd = 'authentication-mode %s' % first_authen_mode
        cmds.append(cmd)
        return cmds

    def merge_authentication_domain(self, **kwargs):
        """ Merge domain of authentication """
        domain_name = kwargs['domain_name']
        authen_scheme_name = kwargs['authen_scheme_name']
        module = kwargs['module']
        conf_str = CE_MERGE_AUTHENTICATION_DOMAIN % (domain_name, authen_scheme_name)
        xml = self.netconf_set_config(module=module, conf_str=conf_str)
        if '<ok/>' not in xml:
            module.fail_json(msg='Error: Merge authentication domain failed.')
        cmds = []
        cmd = 'domain %s' % domain_name
        cmds.append(cmd)
        cmd = 'authentication-scheme %s' % authen_scheme_name
        cmds.append(cmd)
        return cmds

    def create_authentication_scheme(self, **kwargs):
        """ Create scheme of authentication """
        authen_scheme_name = kwargs['authen_scheme_name']
        first_authen_mode = kwargs['first_authen_mode']
        module = kwargs['module']
        conf_str = CE_CREATE_AUTHENTICATION_SCHEME % (authen_scheme_name, first_authen_mode)
        xml = self.netconf_set_config(module=module, conf_str=conf_str)
        if '<ok/>' not in xml:
            module.fail_json(msg='Error: Create authentication scheme failed.')
        cmds = []
        cmd = 'authentication-scheme %s' % authen_scheme_name
        cmds.append(cmd)
        cmd = 'authentication-mode %s' % first_authen_mode
        cmds.append(cmd)
        return cmds

    def create_authentication_domain(self, **kwargs):
        """ Create domain of authentication """
        domain_name = kwargs['domain_name']
        authen_scheme_name = kwargs['authen_scheme_name']
        module = kwargs['module']
        conf_str = CE_CREATE_AUTHENTICATION_DOMAIN % (domain_name, authen_scheme_name)
        xml = self.netconf_set_config(module=module, conf_str=conf_str)
        if '<ok/>' not in xml:
            module.fail_json(msg='Error: Create authentication domain failed.')
        cmds = []
        cmd = 'domain %s' % domain_name
        cmds.append(cmd)
        cmd = 'authentication-scheme %s' % authen_scheme_name
        cmds.append(cmd)
        return cmds

    def delete_authentication_scheme(self, **kwargs):
        """ Delete scheme of authentication """
        authen_scheme_name = kwargs['authen_scheme_name']
        first_authen_mode = kwargs['first_authen_mode']
        module = kwargs['module']
        if authen_scheme_name == 'default':
            return SUCCESS
        conf_str = CE_DELETE_AUTHENTICATION_SCHEME % (authen_scheme_name, first_authen_mode)
        xml = self.netconf_set_config(module=module, conf_str=conf_str)
        if '<ok/>' not in xml:
            module.fail_json(msg='Error: Delete authentication scheme failed.')
        cmds = []
        cmd = 'undo authentication-scheme %s' % authen_scheme_name
        cmds.append(cmd)
        cmd = 'authentication-mode none'
        cmds.append(cmd)
        return cmds

    def delete_authentication_domain(self, **kwargs):
        """ Delete domain of authentication """
        domain_name = kwargs['domain_name']
        authen_scheme_name = kwargs['authen_scheme_name']
        module = kwargs['module']
        if domain_name == 'default':
            return SUCCESS
        conf_str = CE_DELETE_AUTHENTICATION_DOMAIN % (domain_name, authen_scheme_name)
        xml = self.netconf_set_config(module=module, conf_str=conf_str)
        if '<ok/>' not in xml:
            module.fail_json(msg='Error: Delete authentication domain failed.')
        cmds = []
        cmd = 'undo authentication-scheme'
        cmds.append(cmd)
        cmd = 'undo domain %s' % domain_name
        cmds.append(cmd)
        return cmds

    def get_authorization_scheme(self, **kwargs):
        """ Get scheme of authorization """
        module = kwargs['module']
        conf_str = CE_GET_AUTHORIZATION_SCHEME
        xml_str = self.netconf_get_config(module=module, conf_str=conf_str)
        result = list()
        if '<data/>' in xml_str:
            return result
        else:
            re_find = re.findall('.*<authorSchemeName>(.*)</authorSchemeName>.*\\s*<firstAuthorMode>(.*)</firstAuthorMode>.*\\s*<secondAuthorMode>(.*)</secondAuthorMode>.*\\s*', xml_str)
            if re_find:
                return re_find
            else:
                return result

    def get_authorization_domain(self, **kwargs):
        """ Get domain of authorization """
        module = kwargs['module']
        conf_str = CE_GET_AUTHORIZATION_DOMAIN
        xml_str = self.netconf_get_config(module=module, conf_str=conf_str)
        result = list()
        if '<data/>' in xml_str:
            return result
        else:
            re_find = re.findall('.*<domainName>(.*)</domainName>.*\\s*<authorSchemeName>(.*)</authorSchemeName>.*', xml_str)
            if re_find:
                return re_find
            else:
                return result

    def merge_authorization_scheme(self, **kwargs):
        """ Merge scheme of authorization """
        author_scheme_name = kwargs['author_scheme_name']
        first_author_mode = kwargs['first_author_mode']
        module = kwargs['module']
        conf_str = CE_MERGE_AUTHORIZATION_SCHEME % (author_scheme_name, first_author_mode)
        xml = self.netconf_set_config(module=module, conf_str=conf_str)
        if '<ok/>' not in xml:
            module.fail_json(msg='Error: Merge authorization scheme failed.')
        cmds = []
        cmd = 'authorization-scheme %s' % author_scheme_name
        cmds.append(cmd)
        cmd = 'authorization-mode %s' % first_author_mode
        cmds.append(cmd)
        return cmds

    def merge_authorization_domain(self, **kwargs):
        """ Merge domain of authorization """
        domain_name = kwargs['domain_name']
        author_scheme_name = kwargs['author_scheme_name']
        module = kwargs['module']
        conf_str = CE_MERGE_AUTHORIZATION_DOMAIN % (domain_name, author_scheme_name)
        xml = self.netconf_set_config(module=module, conf_str=conf_str)
        if '<ok/>' not in xml:
            module.fail_json(msg='Error: Merge authorization domain failed.')
        cmds = []
        cmd = 'domain %s' % domain_name
        cmds.append(cmd)
        cmd = 'authorization-scheme %s' % author_scheme_name
        cmds.append(cmd)
        return cmds

    def create_authorization_scheme(self, **kwargs):
        """ Create scheme of authorization """
        author_scheme_name = kwargs['author_scheme_name']
        first_author_mode = kwargs['first_author_mode']
        module = kwargs['module']
        conf_str = CE_CREATE_AUTHORIZATION_SCHEME % (author_scheme_name, first_author_mode)
        xml = self.netconf_set_config(module=module, conf_str=conf_str)
        if '<ok/>' not in xml:
            module.fail_json(msg='Error: Create authorization scheme failed.')
        cmds = []
        cmd = 'authorization-scheme %s' % author_scheme_name
        cmds.append(cmd)
        cmd = 'authorization-mode %s' % first_author_mode
        cmds.append(cmd)
        return cmds

    def create_authorization_domain(self, **kwargs):
        """ Create domain of authorization """
        domain_name = kwargs['domain_name']
        author_scheme_name = kwargs['author_scheme_name']
        module = kwargs['module']
        conf_str = CE_CREATE_AUTHORIZATION_DOMAIN % (domain_name, author_scheme_name)
        xml = self.netconf_set_config(module=module, conf_str=conf_str)
        if '<ok/>' not in xml:
            module.fail_json(msg='Error: Create authorization domain failed.')
        cmds = []
        cmd = 'domain %s' % domain_name
        cmds.append(cmd)
        cmd = 'authorization-scheme %s' % author_scheme_name
        cmds.append(cmd)
        return cmds

    def delete_authorization_scheme(self, **kwargs):
        """ Delete scheme of authorization """
        author_scheme_name = kwargs['author_scheme_name']
        first_author_mode = kwargs['first_author_mode']
        module = kwargs['module']
        if author_scheme_name == 'default':
            return SUCCESS
        conf_str = CE_DELETE_AUTHORIZATION_SCHEME % (author_scheme_name, first_author_mode)
        xml = self.netconf_set_config(module=module, conf_str=conf_str)
        if '<ok/>' not in xml:
            module.fail_json(msg='Error: Delete authorization scheme failed.')
        cmds = []
        cmd = 'undo authorization-scheme %s' % author_scheme_name
        cmds.append(cmd)
        cmd = 'authorization-mode none'
        cmds.append(cmd)
        return cmds

    def delete_authorization_domain(self, **kwargs):
        """ Delete domain of authorization """
        domain_name = kwargs['domain_name']
        author_scheme_name = kwargs['author_scheme_name']
        module = kwargs['module']
        if domain_name == 'default':
            return SUCCESS
        conf_str = CE_DELETE_AUTHORIZATION_DOMAIN % (domain_name, author_scheme_name)
        xml = self.netconf_set_config(module=module, conf_str=conf_str)
        if '<ok/>' not in xml:
            module.fail_json(msg='Error: Delete authorization domain failed.')
        cmds = []
        cmd = 'undo authorization-scheme'
        cmds.append(cmd)
        cmd = 'undo domain %s' % domain_name
        cmds.append(cmd)
        return cmds

    def get_accounting_scheme(self, **kwargs):
        """ Get scheme of accounting """
        module = kwargs['module']
        conf_str = CE_GET_ACCOUNTING_SCHEME
        xml_str = self.netconf_get_config(module=module, conf_str=conf_str)
        result = list()
        if '<data/>' in xml_str:
            return result
        else:
            re_find = re.findall('.*<acctSchemeName>(.*)</acctSchemeName>\\s*<accountingMode>(.*)</accountingMode>', xml_str)
            if re_find:
                return re_find
            else:
                return result

    def get_accounting_domain(self, **kwargs):
        """ Get domain of accounting """
        module = kwargs['module']
        conf_str = CE_GET_ACCOUNTING_DOMAIN
        xml_str = self.netconf_get_config(module=module, conf_str=conf_str)
        result = list()
        if '<data/>' in xml_str:
            return result
        else:
            re_find = re.findall('.*<domainName>(.*)</domainName>.*\\s*<acctSchemeName>(.*)</acctSchemeName>.*', xml_str)
            if re_find:
                return re_find
            else:
                return result

    def merge_accounting_scheme(self, **kwargs):
        """ Merge scheme of accounting """
        acct_scheme_name = kwargs['acct_scheme_name']
        accounting_mode = kwargs['accounting_mode']
        module = kwargs['module']
        conf_str = CE_MERGE_ACCOUNTING_SCHEME % (acct_scheme_name, accounting_mode)
        xml = self.netconf_set_config(module=module, conf_str=conf_str)
        if '<ok/>' not in xml:
            module.fail_json(msg='Error: Merge accounting scheme failed.')
        cmds = []
        cmd = 'accounting-scheme %s' % acct_scheme_name
        cmds.append(cmd)
        cmd = 'accounting-mode %s' % accounting_mode
        cmds.append(cmd)
        return cmds

    def merge_accounting_domain(self, **kwargs):
        """ Merge domain of accounting """
        domain_name = kwargs['domain_name']
        acct_scheme_name = kwargs['acct_scheme_name']
        module = kwargs['module']
        conf_str = CE_MERGE_ACCOUNTING_DOMAIN % (domain_name, acct_scheme_name)
        xml = self.netconf_set_config(module=module, conf_str=conf_str)
        if '<ok/>' not in xml:
            module.fail_json(msg='Error: Merge accounting domain failed.')
        cmds = []
        cmd = 'domain %s' % domain_name
        cmds.append(cmd)
        cmd = 'accounting-scheme %s' % acct_scheme_name
        cmds.append(cmd)
        return cmds

    def create_accounting_scheme(self, **kwargs):
        """ Create scheme of accounting """
        acct_scheme_name = kwargs['acct_scheme_name']
        accounting_mode = kwargs['accounting_mode']
        module = kwargs['module']
        conf_str = CE_CREATE_ACCOUNTING_SCHEME % (acct_scheme_name, accounting_mode)
        xml = self.netconf_set_config(module=module, conf_str=conf_str)
        if '<ok/>' not in xml:
            module.fail_json(msg='Error: Create accounting scheme failed.')
        cmds = []
        cmd = 'accounting-scheme %s' % acct_scheme_name
        cmds.append(cmd)
        cmd = 'accounting-mode %s' % accounting_mode
        cmds.append(cmd)
        return cmds

    def create_accounting_domain(self, **kwargs):
        """ Create domain of accounting """
        domain_name = kwargs['domain_name']
        acct_scheme_name = kwargs['acct_scheme_name']
        module = kwargs['module']
        conf_str = CE_CREATE_ACCOUNTING_DOMAIN % (domain_name, acct_scheme_name)
        xml = self.netconf_set_config(module=module, conf_str=conf_str)
        if '<ok/>' not in xml:
            module.fail_json(msg='Error: Create accounting domain failed.')
        cmds = []
        cmd = 'domain %s' % domain_name
        cmds.append(cmd)
        cmd = 'accounting-scheme %s' % acct_scheme_name
        cmds.append(cmd)
        return cmds

    def delete_accounting_scheme(self, **kwargs):
        """ Delete scheme of accounting """
        acct_scheme_name = kwargs['acct_scheme_name']
        accounting_mode = kwargs['accounting_mode']
        module = kwargs['module']
        if acct_scheme_name == 'default':
            return SUCCESS
        conf_str = CE_DELETE_ACCOUNTING_SCHEME % (acct_scheme_name, accounting_mode)
        xml = self.netconf_set_config(module=module, conf_str=conf_str)
        if '<ok/>' not in xml:
            module.fail_json(msg='Error: Delete accounting scheme failed.')
        cmds = []
        cmd = 'undo accounting-scheme %s' % acct_scheme_name
        cmds.append(cmd)
        cmd = 'accounting-mode none'
        cmds.append(cmd)
        return cmds

    def delete_accounting_domain(self, **kwargs):
        """ Delete domain of accounting """
        domain_name = kwargs['domain_name']
        acct_scheme_name = kwargs['acct_scheme_name']
        module = kwargs['module']
        if domain_name == 'default':
            return SUCCESS
        conf_str = CE_DELETE_ACCOUNTING_DOMAIN % (domain_name, acct_scheme_name)
        xml = self.netconf_set_config(module=module, conf_str=conf_str)
        if '<ok/>' not in xml:
            module.fail_json(msg='Error: Delete accounting domain failed.')
        cmds = []
        cmd = 'undo domain %s' % domain_name
        cmds.append(cmd)
        cmd = 'undo accounting-scheme'
        cmds.append(cmd)
        return cmds

    def get_radius_template(self, **kwargs):
        """ Get radius template """
        module = kwargs['module']
        conf_str = CE_GET_RADIUS_TEMPLATE
        xml_str = self.netconf_get_config(module=module, conf_str=conf_str)
        result = list()
        if '<data/>' in xml_str:
            return result
        else:
            re_find = re.findall('.*<groupName>(.*)</groupName>.*', xml_str)
            if re_find:
                return re_find
            else:
                return result

    def merge_radius_template(self, **kwargs):
        """ Merge radius template """
        radius_server_group = kwargs['radius_server_group']
        module = kwargs['module']
        conf_str = CE_MERGE_RADIUS_TEMPLATE % radius_server_group
        xml = self.netconf_set_config(module=module, conf_str=conf_str)
        if '<ok/>' not in xml:
            module.fail_json(msg='Error: Merge radius template failed.')
        cmds = []
        cmd = 'radius server group %s' % radius_server_group
        cmds.append(cmd)
        return cmds

    def create_radius_template(self, **kwargs):
        """ Create radius template """
        radius_server_group = kwargs['radius_server_group']
        module = kwargs['module']
        conf_str = CE_CREATE_RADIUS_TEMPLATE % radius_server_group
        xml = self.netconf_set_config(module=module, conf_str=conf_str)
        if '<ok/>' not in xml:
            module.fail_json(msg='Error: Create radius template failed.')
        cmds = []
        cmd = 'radius server group %s' % radius_server_group
        cmds.append(cmd)
        return cmds

    def delete_radius_template(self, **kwargs):
        """ Delete radius template """
        radius_server_group = kwargs['radius_server_group']
        module = kwargs['module']
        conf_str = CE_DELETE_RADIUS_TEMPLATE % radius_server_group
        xml = self.netconf_set_config(module=module, conf_str=conf_str)
        if '<ok/>' not in xml:
            module.fail_json(msg='Error: Delete radius template failed.')
        cmds = []
        cmd = 'undo radius server group %s' % radius_server_group
        cmds.append(cmd)
        return cmds

    def get_radius_client(self, **kwargs):
        """ Get radius client """
        module = kwargs['module']
        conf_str = CE_GET_RADIUS_CLIENT
        xml_str = self.netconf_get_config(module=module, conf_str=conf_str)
        result = list()
        if '<data/>' in xml_str:
            return result
        else:
            re_find = re.findall('.*<isEnable>(.*)</isEnable>.*', xml_str)
            if re_find:
                return re_find
            else:
                return result

    def merge_radius_client(self, **kwargs):
        """ Merge radius client """
        enable = kwargs['isEnable']
        module = kwargs['module']
        conf_str = CE_MERGE_RADIUS_CLIENT % enable
        xml = self.netconf_set_config(module=module, conf_str=conf_str)
        if '<ok/>' not in xml:
            module.fail_json(msg='Error: Merge radius client failed.')
        cmds = []
        if enable == 'true':
            cmd = 'radius enable'
        else:
            cmd = 'undo radius enable'
        cmds.append(cmd)
        return cmds

    def get_hwtacacs_template(self, **kwargs):
        """ Get hwtacacs template """
        module = kwargs['module']
        conf_str = CE_GET_HWTACACS_TEMPLATE
        xml_str = self.netconf_get_config(module=module, conf_str=conf_str)
        result = list()
        if '<data/>' in xml_str:
            return result
        else:
            re_find = re.findall('.*<templateName>(.*)</templateName>.*', xml_str)
            if re_find:
                return re_find
            else:
                return result

    def merge_hwtacacs_template(self, **kwargs):
        """ Merge hwtacacs template """
        hwtacas_template = kwargs['hwtacas_template']
        module = kwargs['module']
        conf_str = CE_MERGE_HWTACACS_TEMPLATE % hwtacas_template
        xml = self.netconf_set_config(module=module, conf_str=conf_str)
        if '<ok/>' not in xml:
            module.fail_json(msg='Error: Merge hwtacacs template failed.')
        cmds = []
        cmd = 'hwtacacs server template %s' % hwtacas_template
        cmds.append(cmd)
        return cmds

    def create_hwtacacs_template(self, **kwargs):
        """ Create hwtacacs template """
        hwtacas_template = kwargs['hwtacas_template']
        module = kwargs['module']
        conf_str = CE_CREATE_HWTACACS_TEMPLATE % hwtacas_template
        xml = self.netconf_set_config(module=module, conf_str=conf_str)
        if '<ok/>' not in xml:
            module.fail_json(msg='Error: Create hwtacacs template failed.')
        cmds = []
        cmd = 'hwtacacs server template %s' % hwtacas_template
        cmds.append(cmd)
        return cmds

    def delete_hwtacacs_template(self, **kwargs):
        """ Delete hwtacacs template """
        hwtacas_template = kwargs['hwtacas_template']
        module = kwargs['module']
        conf_str = CE_DELETE_HWTACACS_TEMPLATE % hwtacas_template
        xml = self.netconf_set_config(module=module, conf_str=conf_str)
        if '<ok/>' not in xml:
            module.fail_json(msg='Error: Delete hwtacacs template failed.')
        cmds = []
        cmd = 'undo hwtacacs server template %s' % hwtacas_template
        cmds.append(cmd)
        return cmds

    def get_hwtacacs_global_cfg(self, **kwargs):
        """ Get hwtacacs global configure """
        module = kwargs['module']
        conf_str = CE_GET_HWTACACS_GLOBAL_CFG
        xml_str = self.netconf_get_config(module=module, conf_str=conf_str)
        result = list()
        if '<data/>' in xml_str:
            return result
        else:
            re_find = re.findall('.*<isEnable>(.*)</isEnable>.*', xml_str)
            if re_find:
                return re_find
            else:
                return result

    def merge_hwtacacs_global_cfg(self, **kwargs):
        """ Merge hwtacacs global configure """
        enable = kwargs['isEnable']
        module = kwargs['module']
        conf_str = CE_MERGE_HWTACACS_GLOBAL_CFG % enable
        xml = self.netconf_set_config(module=module, conf_str=conf_str)
        if '<ok/>' not in xml:
            module.fail_json(msg='Error: Merge hwtacacs global config failed.')
        cmds = []
        if enable == 'true':
            cmd = 'hwtacacs enable'
        else:
            cmd = 'undo hwtacacs enable'
        cmds.append(cmd)
        return cmds

    def get_local_user_group(self, **kwargs):
        """ Get local user group """
        module = kwargs['module']
        conf_str = CE_GET_LOCAL_USER_GROUP
        xml_str = self.netconf_get_config(module=module, conf_str=conf_str)
        result = list()
        if '<data/>' in xml_str:
            return result
        else:
            re_find = re.findall('.*<userGroupName>(.*)</userGroupName>.*', xml_str)
            if re_find:
                return re_find
            else:
                return result

    def merge_local_user_group(self, **kwargs):
        """ Merge local user group """
        local_user_group = kwargs['local_user_group']
        module = kwargs['module']
        conf_str = CE_MERGE_LOCAL_USER_GROUP % local_user_group
        xml = self.netconf_set_config(module=module, conf_str=conf_str)
        if '<ok/>' not in xml:
            module.fail_json(msg='Error: Merge local user group failed.')
        cmds = []
        cmd = 'user-group %s' % local_user_group
        cmds.append(cmd)
        return cmds

    def delete_local_user_group(self, **kwargs):
        """ Delete local user group """
        local_user_group = kwargs['local_user_group']
        module = kwargs['module']
        conf_str = CE_DELETE_LOCAL_USER_GROUP % local_user_group
        xml = self.netconf_set_config(module=module, conf_str=conf_str)
        if '<ok/>' not in xml:
            module.fail_json(msg='Error: Delete local user group failed.')
        cmds = []
        cmd = 'undo user-group %s' % local_user_group
        cmds.append(cmd)
        return cmds