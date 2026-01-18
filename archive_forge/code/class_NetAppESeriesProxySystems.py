from __future__ import absolute_import, division, print_function
import json
import threading
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
from ansible.module_utils._text import to_native
from time import sleep
class NetAppESeriesProxySystems(NetAppESeriesModule):
    DEFAULT_CONNECTION_TIMEOUT_SEC = 30
    DEFAULT_GRAPH_DISCOVERY_TIMEOUT = 30
    DEFAULT_PASSWORD_STATE_TIMEOUT = 30
    DEFAULT_DISCOVERY_TIMEOUT_SEC = 300

    def __init__(self):
        ansible_options = dict(add_discovered_systems=dict(type='bool', required=False, default=False), subnet_mask=dict(type='str', required=False), password=dict(type='str', required=False, default='', no_log=True), tags=dict(type='dict', required=False), accept_certificate=dict(type='bool', required=False, default=True), systems=dict(type='list', required=False, default=[], suboptions=dict(ssid=dict(type='str', required=False), serial=dict(type='str', required=False), addresses=dict(type='list', required=False), password=dict(type='str', required=False, no_log=True), tags=dict(type='dict', required=False))))
        super(NetAppESeriesProxySystems, self).__init__(ansible_options=ansible_options, web_services_version='04.10.0000.0000', supports_check_mode=True, proxy_specific_task=True)
        args = self.module.params
        self.add_discovered_systems = args['add_discovered_systems']
        self.subnet_mask = args['subnet_mask']
        self.accept_certificate = args['accept_certificate']
        self.default_password = args['password']
        self.default_meta_tags = []
        if 'tags' in args and args['tags']:
            for key in args['tags'].keys():
                if isinstance(args['tags'][key], list):
                    self.default_meta_tags.append({'key': key, 'valueList': args['tags'][key]})
                else:
                    self.default_meta_tags.append({'key': key, 'valueList': [args['tags'][key]]})
        self.default_meta_tags = sorted(self.default_meta_tags, key=lambda x: x['key'])
        self.undiscovered_systems = []
        self.systems_to_remove = []
        self.systems_to_update = []
        self.systems_to_add = []
        self.serial_numbers = []
        self.systems = []
        if args['systems']:
            for system in args['systems']:
                if isinstance(system, str):
                    self.serial_numbers.append(system)
                    self.systems.append({'ssid': system, 'serial': system, 'password': self.default_password, 'password_valid': None, 'password_set': None, 'stored_password_valid': None, 'meta_tags': self.default_meta_tags, 'controller_addresses': [], 'embedded_available': None, 'accept_certificate': False, 'current_info': {}, 'changes': {}, 'updated_required': False, 'failed': False, 'discovered': False})
                elif isinstance(system, dict):
                    if 'ssid' not in system:
                        if 'serial' in system and system['serial']:
                            system.update({'ssid': system['serial']})
                        elif 'addresses' in system and system['addresses']:
                            system.update({'ssid': system['addresses'][0]})
                    if 'password' not in system:
                        system.update({'password': self.default_password})
                    if 'serial' in system and system['serial']:
                        self.serial_numbers.append(system['serial'])
                    meta_tags = self.default_meta_tags
                    if 'meta_tags' in system and system['meta_tags']:
                        for key in system['meta_tags'].keys():
                            if isinstance(system['meta_tags'][key], list):
                                meta_tags.append({'key': key, 'valueList': system['meta_tags'][key]})
                            else:
                                meta_tags.append({'key': key, 'valueList': [system['meta_tags'][key]]})
                        meta_tags = sorted(meta_tags, key=lambda x: x['key'])
                    self.systems.append({'ssid': str(system['ssid']), 'serial': system['serial'] if 'serial' in system else '', 'password': system['password'], 'password_valid': None, 'password_set': None, 'stored_password_valid': None, 'meta_tags': meta_tags, 'controller_addresses': system['addresses'] if 'addresses' in system else [], 'embedded_available': None, 'accept_certificate': False, 'current_info': {}, 'changes': {}, 'updated_required': False, 'failed': False, 'discovered': False})
                else:
                    self.module.fail_json(msg='Invalid system! All systems must either be a simple serial number or a dictionary. Failed system: %s' % system)
        self.DEFAULT_HEADERS.update({'x-netapp-password-validate-method': 'none'})

    def discover_array(self):
        """Search for array using the world wide identifier."""
        subnet = ipaddress.ip_network(u'%s' % self.subnet_mask)
        try:
            rc, request_id = self.request('discovery', method='POST', data={'startIP': str(subnet[0]), 'endIP': str(subnet[-1]), 'connectionTimeout': self.DEFAULT_CONNECTION_TIMEOUT_SEC})
            discovered_systems = None
            try:
                for iteration in range(self.DEFAULT_DISCOVERY_TIMEOUT_SEC):
                    rc, discovered_systems = self.request('discovery?requestId=%s' % request_id['requestId'])
                    if not discovered_systems['discoverProcessRunning']:
                        break
                    sleep(1)
                else:
                    self.module.fail_json(msg='Timeout waiting for array discovery process. Subnet [%s]' % self.subnet_mask)
            except Exception as error:
                self.module.fail_json(msg='Failed to get the discovery results. Error [%s].' % to_native(error))
            if not discovered_systems:
                self.module.warn('Discovery found no systems. IP starting address [%s]. IP ending address: [%s].' % (str(subnet[0]), str(subnet[-1])))
            else:
                if self.add_discovered_systems:
                    for discovered_system in discovered_systems['storageSystems']:
                        if discovered_system['serialNumber'] not in self.serial_numbers:
                            self.systems.append({'ssid': discovered_system['serialNumber'], 'serial': discovered_system['serialNumber'], 'password': self.default_password, 'password_valid': None, 'password_set': None, 'stored_password_valid': None, 'meta_tags': self.default_meta_tags, 'controller_addresses': [], 'embedded_available': None, 'accept_certificate': False, 'current_info': {}, 'changes': {}, 'updated_required': False, 'failed': False, 'discovered': False})
                for system in self.systems:
                    for discovered_system in discovered_systems['storageSystems']:
                        if system['serial'] == discovered_system['serialNumber'] or (system['controller_addresses'] and all([address in discovered_system['ipAddresses'] for address in system['controller_addresses']])):
                            system['controller_addresses'] = sorted(discovered_system['ipAddresses'])
                            system['embedded_available'] = 'https' in discovered_system['supportedManagementPorts']
                            system['accept_certificate'] = system['embedded_available'] and self.accept_certificate
                            system['discovered'] = True
                            break
                    else:
                        self.undiscovered_systems.append(system['ssid'])
        except Exception as error:
            self.module.fail_json(msg='Failed to initiate array discovery. Error [%s].' % to_native(error))

    def update_storage_systems_info(self):
        """Get current web services proxy storage systems."""
        try:
            rc, existing_systems = self.request('storage-systems')
            for system in self.systems:
                for existing_system in existing_systems:
                    if system['ssid'] == existing_system['id']:
                        system['current_info'] = existing_system
                        if system['current_info']['passwordStatus'] in ['unknown', 'securityLockout']:
                            system['failed'] = True
                            self.module.warn('Skipping storage system [%s] because of current password status [%s]' % (system['ssid'], system['current_info']['passwordStatus']))
                        if system['current_info']['metaTags']:
                            system['current_info']['metaTags'] = sorted(system['current_info']['metaTags'], key=lambda x: x['key'])
                        break
                else:
                    self.systems_to_add.append(system)
            for existing_system in existing_systems:
                for system in self.systems:
                    if existing_system['id'] == system['ssid']:
                        if existing_system['id'] in self.undiscovered_systems:
                            self.undiscovered_systems.remove(existing_system['id'])
                            self.module.warn('Expected storage system exists on the proxy but was failed to be discovered. Array [%s].' % existing_system['id'])
                        break
                else:
                    self.systems_to_remove.append(existing_system['id'])
        except Exception as error:
            self.module.fail_json(msg='Failed to retrieve storage systems. Error [%s].' % to_native(error))

    def set_password(self, system):
        """Determine whether password has been set and, if it hasn't been set, set it."""
        if system['embedded_available'] and system['controller_addresses']:
            for url in ['https://%s:8443/devmgr' % system['controller_addresses'][0], 'https://%s:443/devmgr' % system['controller_addresses'][0], 'http://%s:8080/devmgr' % system['controller_addresses'][0]]:
                try:
                    rc, response = self._request('%s/utils/login?uid=admin&xsrf=false&onlycheck=true' % url, ignore_errors=True, url_username='admin', url_password='', validate_certs=False)
                    if rc == 200:
                        system['password_set'] = False
                        if system['password']:
                            try:
                                rc, storage_system = self._request('%s/v2/storage-systems/1/passwords' % url, method='POST', url_username='admin', headers=self.DEFAULT_HEADERS, url_password='', validate_certs=False, data=json.dumps({'currentAdminPassword': '', 'adminPassword': True, 'newPassword': system['password']}))
                            except Exception as error:
                                system['failed'] = True
                                self.module.warn('Failed to set storage system password. Array [%s].' % system['ssid'])
                        break
                    elif rc == 401:
                        system['password_set'] = True
                        break
                except Exception as error:
                    pass
            else:
                self.module.warn('Failed to retrieve array password state. Array [%s].' % system['ssid'])
                system['failed'] = True

    def update_system_changes(self, system):
        """Determine whether storage system configuration changes are required """
        if system['current_info']:
            system['changes'] = dict()
            if sorted(system['controller_addresses']) != sorted(system['current_info']['managementPaths']) or system['current_info']['ip1'] not in system['current_info']['managementPaths'] or system['current_info']['ip2'] not in system['current_info']['managementPaths']:
                system['changes'].update({'controllerAddresses': system['controller_addresses']})
            if len(system['meta_tags']) != len(system['current_info']['metaTags']):
                if len(system['meta_tags']) == 0:
                    system['changes'].update({'removeAllTags': True})
                else:
                    system['changes'].update({'metaTags': system['meta_tags']})
            else:
                for index in range(len(system['meta_tags'])):
                    if system['current_info']['metaTags'][index]['key'] != system['meta_tags'][index]['key'] or sorted(system['current_info']['metaTags'][index]['valueList']) != sorted(system['meta_tags'][index]['valueList']):
                        system['changes'].update({'metaTags': system['meta_tags']})
                        break
            if system['accept_certificate'] and (not all([controller['certificateStatus'] == 'trusted' for controller in system['current_info']['controllers']])):
                system['changes'].update({'acceptCertificate': True})
        if system['id'] not in self.undiscovered_systems and system['changes']:
            self.systems_to_update.append(system)

    def add_system(self, system):
        """Add basic storage system definition to the web services proxy."""
        self.set_password(system)
        body = {'id': system['ssid'], 'controllerAddresses': system['controller_addresses'], 'password': system['password']}
        if system['accept_certificate']:
            body.update({'acceptCertificate': system['accept_certificate']})
        if system['meta_tags']:
            body.update({'metaTags': system['meta_tags']})
        try:
            rc, storage_system = self.request('storage-systems', method='POST', data=body)
        except Exception as error:
            self.module.warn('Failed to add storage system. Array [%s]. Error [%s]' % (system['ssid'], to_native(error)))
            return
        for retries in range(5):
            sleep(1)
            try:
                rc, storage_system = self.request('storage-systems/%s/validatePassword' % system['ssid'], method='POST')
                break
            except Exception as error:
                continue
        else:
            self.module.warn('Failed to validate password status. Array [%s]. Error [%s]' % (system['ssid'], to_native(error)))

    def update_system(self, system):
        """Update storage system configuration."""
        try:
            rc, storage_system = self.request('storage-systems/%s' % system['ssid'], method='POST', data=system['changes'])
        except Exception as error:
            self.module.warn('Failed to update storage system. Array [%s]. Error [%s]' % (system['ssid'], to_native(error)))

    def remove_system(self, ssid):
        """Remove storage system."""
        try:
            rc, storage_system = self.request('storage-systems/%s' % ssid, method='DELETE')
        except Exception as error:
            self.module.warn('Failed to remove storage system. Array [%s]. Error [%s].' % (ssid, to_native(error)))

    def apply(self):
        """Determine whether changes are required and, if necessary, apply them."""
        missing_packages = []
        if not HAS_IPADDRESS:
            missing_packages.append('ipaddress')
        if missing_packages:
            self.module.fail_json(msg='Python packages are missing! Packages [%s].' % ', '.join(missing_packages))
        if self.is_embedded():
            self.module.fail_json(msg='Cannot add/remove storage systems to SANtricity Web Services Embedded instance.')
        if self.add_discovered_systems or self.systems:
            if self.subnet_mask:
                self.discover_array()
            self.update_storage_systems_info()
            thread_pool = []
            for system in self.systems:
                if not system['failed']:
                    thread = threading.Thread(target=self.update_system_changes, args=(system,))
                    thread_pool.append(thread)
                    thread.start()
            for thread in thread_pool:
                thread.join()
        else:
            self.update_storage_systems_info()
        changes_required = False
        if self.systems_to_add or self.systems_to_update or self.systems_to_remove:
            changes_required = True
        if changes_required and (not self.module.check_mode):
            add_msg = ''
            update_msg = ''
            remove_msg = ''
            if self.systems_to_remove:
                ssids = []
                thread_pool = []
                for ssid in self.systems_to_remove:
                    thread = threading.Thread(target=self.remove_system, args=(ssid,))
                    thread_pool.append(thread)
                    thread.start()
                    ssids.append(ssid)
                for thread in thread_pool:
                    thread.join()
                if ssids:
                    remove_msg = 'system%s removed: %s' % ('s' if len(ssids) > 1 else '', ', '.join(ssids))
            thread_pool = []
            if self.systems_to_add:
                ssids = []
                for system in self.systems_to_add:
                    if not system['failed']:
                        thread = threading.Thread(target=self.add_system, args=(system,))
                        thread_pool.append(thread)
                        thread.start()
                        ssids.append(system['ssid'])
                if ssids:
                    add_msg = 'system%s added: %s' % ('s' if len(ssids) > 1 else '', ', '.join(ssids))
            if self.systems_to_update:
                ssids = []
                for system in self.systems_to_update:
                    if not system['failed']:
                        thread = threading.Thread(target=self.update_system, args=(system,))
                        thread_pool.append(thread)
                        thread.start()
                        ssids.append(system['ssid'])
                if ssids:
                    update_msg = 'system%s updated: %s' % ('s' if len(ssids) > 1 else '', ', '.join(ssids))
            for thread in thread_pool:
                thread.join()
            if self.undiscovered_systems:
                undiscovered_msg = 'system%s undiscovered: %s' % ('s ' if len(self.undiscovered_systems) > 1 else '', ', '.join(self.undiscovered_systems))
                self.module.fail_json(msg=', '.join([msg for msg in [add_msg, update_msg, remove_msg, undiscovered_msg] if msg]), changed=changes_required)
            self.module.exit_json(msg=', '.join([msg for msg in [add_msg, update_msg, remove_msg] if msg]), changed=changes_required)
        if self.undiscovered_systems:
            self.module.fail_json(msg='No changes were made; however the following system(s) failed to be discovered: %s.' % self.undiscovered_systems, changed=changes_required)
        self.module.exit_json(msg='No changes were made.', changed=changes_required)