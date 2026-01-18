from __future__ import absolute_import, division, print_function
import json
import logging
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import request, eseries_host_argument_spec
from ansible.module_utils._text import to_native
class Ldap(object):
    NO_CHANGE_MSG = 'No changes were necessary.'

    def __init__(self):
        argument_spec = eseries_host_argument_spec()
        argument_spec.update(dict(state=dict(type='str', required=False, default='present', choices=['present', 'absent']), identifier=dict(type='str', required=False), username=dict(type='str', required=False, aliases=['bind_username']), password=dict(type='str', required=False, aliases=['bind_password'], no_log=True), name=dict(type='list', required=False), server=dict(type='str', required=False, aliases=['server_url']), search_base=dict(type='str', required=False), role_mappings=dict(type='dict', required=False), user_attribute=dict(type='str', required=False, default='sAMAccountName'), attributes=dict(type='list', default=['memberOf'], required=False), log_path=dict(type='str', required=False)))
        required_if = [['state', 'present', ['username', 'password', 'server', 'search_base', 'role_mappings']]]
        self.module = AnsibleModule(argument_spec=argument_spec, supports_check_mode=True, required_if=required_if)
        args = self.module.params
        self.ldap = args['state'] == 'present'
        self.identifier = args['identifier']
        self.username = args['username']
        self.password = args['password']
        self.names = args['name']
        self.server = args['server']
        self.search_base = args['search_base']
        self.role_mappings = args['role_mappings']
        self.user_attribute = args['user_attribute']
        self.attributes = args['attributes']
        self.ssid = args['ssid']
        self.url = args['api_url']
        self.creds = dict(url_password=args['api_password'], validate_certs=args['validate_certs'], url_username=args['api_username'], timeout=60)
        self.check_mode = self.module.check_mode
        log_path = args['log_path']
        self._logger = logging.getLogger(self.__class__.__name__)
        if log_path:
            logging.basicConfig(level=logging.DEBUG, filename=log_path, filemode='w', format='%(relativeCreated)dms %(levelname)s %(module)s.%(funcName)s:%(lineno)d\n %(message)s')
        if not self.url.endswith('/'):
            self.url += '/'
        self.embedded = None
        self.base_path = None

    def make_configuration(self):
        if not self.identifier:
            self.identifier = 'default'
        if not self.names:
            parts = urlparse.urlparse(self.server)
            netloc = parts.netloc
            if ':' in netloc:
                netloc = netloc.split(':')[0]
            self.names = [netloc]
        roles = list()
        for regex in self.role_mappings:
            for role in self.role_mappings[regex]:
                roles.append(dict(groupRegex=regex, ignoreCase=True, name=role))
        domain = dict(id=self.identifier, ldapUrl=self.server, bindLookupUser=dict(user=self.username, password=self.password), roleMapCollection=roles, groupAttributes=self.attributes, names=self.names, searchBase=self.search_base, userAttribute=self.user_attribute)
        return domain

    def is_embedded(self):
        """Determine whether or not we're using the embedded or proxy implementation of Web Services"""
        if self.embedded is None:
            url = self.url
            try:
                parts = urlparse.urlparse(url)
                parts = parts._replace(path='/devmgr/utils/')
                url = urlparse.urlunparse(parts)
                rc, result = request(url + 'about', **self.creds)
                self.embedded = not result['runningAsProxy']
            except Exception as err:
                self._logger.exception('Failed to retrieve the About information.')
                self.module.fail_json(msg='Failed to determine the Web Services implementation type! Array Id [%s]. Error [%s].' % (self.ssid, to_native(err)))
        return self.embedded

    def get_full_configuration(self):
        try:
            rc, result = request(self.url + self.base_path, **self.creds)
            return result
        except Exception as err:
            self._logger.exception('Failed to retrieve the LDAP configuration.')
            self.module.fail_json(msg='Failed to retrieve LDAP configuration! Array Id [%s]. Error [%s].' % (self.ssid, to_native(err)))

    def get_configuration(self, identifier):
        try:
            rc, result = request(self.url + self.base_path + '%s' % identifier, ignore_errors=True, **self.creds)
            if rc == 200:
                return result
            elif rc == 404:
                return None
            else:
                self.module.fail_json(msg='Failed to retrieve LDAP configuration! Array Id [%s]. Error [%s].' % (self.ssid, result))
        except Exception as err:
            self._logger.exception('Failed to retrieve the LDAP configuration.')
            self.module.fail_json(msg='Failed to retrieve LDAP configuration! Array Id [%s]. Error [%s].' % (self.ssid, to_native(err)))

    def update_configuration(self):
        domain = self.make_configuration()
        current = self.get_configuration(self.identifier)
        update = current != domain
        msg = 'No changes were necessary for [%s].' % self.identifier
        self._logger.info('Is updated: %s', update)
        if update and (not self.check_mode):
            msg = 'The configuration changes were made for [%s].' % self.identifier
            try:
                if current is None:
                    api = self.base_path + 'addDomain'
                else:
                    api = self.base_path + '%s' % domain['id']
                rc, result = request(self.url + api, method='POST', data=json.dumps(domain), **self.creds)
            except Exception as err:
                self._logger.exception('Failed to modify the LDAP configuration.')
                self.module.fail_json(msg='Failed to modify LDAP configuration! Array Id [%s]. Error [%s].' % (self.ssid, to_native(err)))
        return (msg, update)

    def clear_single_configuration(self, identifier=None):
        if identifier is None:
            identifier = self.identifier
        configuration = self.get_configuration(identifier)
        updated = False
        msg = self.NO_CHANGE_MSG
        if configuration:
            updated = True
            msg = 'The LDAP domain configuration for [%s] was cleared.' % identifier
            if not self.check_mode:
                try:
                    rc, result = request(self.url + self.base_path + '%s' % identifier, method='DELETE', **self.creds)
                except Exception as err:
                    self.module.fail_json(msg='Failed to remove LDAP configuration! Array Id [%s]. Error [%s].' % (self.ssid, to_native(err)))
        return (msg, updated)

    def clear_configuration(self):
        configuration = self.get_full_configuration()
        updated = False
        msg = self.NO_CHANGE_MSG
        if configuration['ldapDomains']:
            updated = True
            msg = 'The LDAP configuration for all domains was cleared.'
            if not self.check_mode:
                try:
                    rc, result = request(self.url + self.base_path, method='DELETE', ignore_errors=True, **self.creds)
                    if rc == 405:
                        for config in configuration['ldapDomains']:
                            self.clear_single_configuration(config['id'])
                except Exception as err:
                    self.module.fail_json(msg='Failed to clear LDAP configuration! Array Id [%s]. Error [%s].' % (self.ssid, to_native(err)))
        return (msg, updated)

    def get_base_path(self):
        embedded = self.is_embedded()
        if embedded:
            return 'storage-systems/%s/ldap/' % self.ssid
        else:
            return '/ldap/'

    def update(self):
        self.base_path = self.get_base_path()
        if self.ldap:
            msg, update = self.update_configuration()
        elif self.identifier:
            msg, update = self.clear_single_configuration()
        else:
            msg, update = self.clear_configuration()
        self.module.exit_json(msg=msg, changed=update)

    def __call__(self, *args, **kwargs):
        self.update()