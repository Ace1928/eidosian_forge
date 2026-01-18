from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils._text import to_native
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
class NetAppESeriesHost(NetAppESeriesModule):
    PORT_TYPES = ['iscsi', 'sas', 'fc', 'ib', 'nvmeof']

    def __init__(self):
        ansible_options = dict(state=dict(type='str', default='present', choices=['absent', 'present']), ports=dict(type='list', required=False), force_port=dict(type='bool', default=False), name=dict(type='str', required=True, aliases=['label']), host_type=dict(type='str', required=False, aliases=['host_type_index']))
        super(NetAppESeriesHost, self).__init__(ansible_options=ansible_options, web_services_version='02.00.0000.0000', supports_check_mode=True)
        self.check_mode = self.module.check_mode
        args = self.module.params
        self.ports = args['ports']
        self.force_port = args['force_port']
        self.name = args['name']
        self.state = args['state']
        self.post_body = dict()
        self.all_hosts = list()
        self.host_obj = dict()
        self.new_ports = list()
        self.ports_for_update = list()
        self.ports_for_removal = list()
        host_type = args['host_type']
        if host_type:
            host_type = host_type.lower()
            if host_type in [key.lower() for key in list(self.HOST_TYPE_INDEXES.keys())]:
                self.host_type_index = self.HOST_TYPE_INDEXES[host_type]
            elif host_type.isdigit():
                self.host_type_index = int(args['host_type'])
            else:
                self.module.fail_json(msg='host_type must be either a host type name or host type index found integer the documentation.')
        else:
            self.host_type_index = None
        if not self.url.endswith('/'):
            self.url += '/'
        if self.ports is not None:
            for port in self.ports:
                port['type'] = port['type'].lower()
                port['port'] = port['port'].lower()
                if port['type'] not in self.PORT_TYPES:
                    self.module.fail_json(msg='Invalid port type! Port interface type must be one of [%s].' % ', '.join(self.PORT_TYPES))
                if re.match('^(0x)?[0-9a-f]{16}$', port['port'].replace(':', '')):
                    port['port'] = port['port'].replace(':', '').replace('0x', '')
                    if port['type'] == 'ib':
                        port['port'] = '0' * (32 - len(port['port'])) + port['port']

    @property
    def default_host_type(self):
        """Return the default host type index."""
        try:
            rc, default_index = self.request('storage-systems/%s/graph/xpath-filter?query=/sa/defaultHostTypeIndex' % self.ssid)
            return default_index[0]
        except Exception as error:
            self.module.fail_json(msg='Failed to retrieve default host type index')

    @property
    def valid_host_type(self):
        host_types = None
        try:
            rc, host_types = self.request('storage-systems/%s/host-types' % self.ssid)
        except Exception as err:
            self.module.fail_json(msg='Failed to get host types. Array Id [%s]. Error [%s].' % (self.ssid, to_native(err)))
        try:
            match = list(filter(lambda host_type: host_type['index'] == self.host_type_index, host_types))[0]
            return True
        except IndexError:
            self.module.fail_json(msg='There is no host type with index %s' % self.host_type_index)

    def check_port_types(self):
        """Check to see whether the port interface types are available on storage system."""
        try:
            rc, interfaces = self.request('storage-systems/%s/interfaces?channelType=hostside' % self.ssid)
            for port in self.ports:
                for interface in interfaces:
                    if port['type'] == 'ib' and 'iqn' in port['port']:
                        if interface['ioInterfaceTypeData']['interfaceType'] == 'iscsi' and interface['ioInterfaceTypeData']['iscsi']['interfaceData']['type'] == 'infiniband' and interface['ioInterfaceTypeData']['iscsi']['interfaceData']['infinibandData']['isIser'] or (interface['ioInterfaceTypeData']['interfaceType'] == 'ib' and interface['ioInterfaceTypeData']['ib']['isISERSupported']):
                            port['type'] = 'iscsi'
                            break
                    elif port['type'] == 'nvmeof' and 'commandProtocolPropertiesList' in interface and ('commandProtocolProperties' in interface['commandProtocolPropertiesList']) and interface['commandProtocolPropertiesList']['commandProtocolProperties']:
                        if interface['commandProtocolPropertiesList']['commandProtocolProperties'][0]['commandProtocol'] == 'nvme':
                            break
                    elif port['type'] == 'fc' and interface['ioInterfaceTypeData']['interfaceType'] == 'fibre' or port['type'] == interface['ioInterfaceTypeData']['interfaceType']:
                        break
                else:
                    self.module.warn('Port type not found in hostside interfaces! Type [%s]. Port [%s].' % (port['type'], port['label']))
        except Exception as error:
            for port in self.ports:
                if port['type'] == 'ib' and 'iqn' in port['port']:
                    port['type'] = 'iscsi'
                    break

    def assigned_host_ports(self, apply_unassigning=False):
        """Determine if the hostPorts requested have already been assigned and return list of required used ports."""
        used_host_ports = {}
        for host in self.all_hosts:
            if host['label'].lower() != self.name.lower():
                for host_port in host['hostSidePorts']:
                    for port in self.ports:
                        if port['port'] == host_port['address'] or port['label'].lower() == host_port['label'].lower():
                            if not self.force_port:
                                self.module.fail_json(msg='Port label or address is already used and force_port option is set to false!')
                            else:
                                port_ref = [port['hostPortRef'] for port in host['ports'] if port['hostPortName'] == host_port['address']]
                                port_ref.extend([port['initiatorRef'] for port in host['initiators'] if port['nodeName']['iscsiNodeName'] == host_port['address']])
                                if host['hostRef'] not in used_host_ports.keys():
                                    used_host_ports.update({host['hostRef']: port_ref})
                                else:
                                    used_host_ports[host['hostRef']].extend(port_ref)
        if apply_unassigning:
            for host_ref in used_host_ports.keys():
                try:
                    rc, resp = self.request('storage-systems/%s/hosts/%s' % (self.ssid, host_ref), method='POST', data={'portsToRemove': used_host_ports[host_ref]})
                except Exception as err:
                    self.module.fail_json(msg='Failed to unassign host port. Host Id [%s]. Array Id [%s]. Ports [%s]. Error [%s].' % (self.host_obj['id'], self.ssid, used_host_ports[host_ref], to_native(err)))

    @property
    def host_exists(self):
        """Determine if the requested host exists
        As a side effect, set the full list of defined hosts in "all_hosts", and the target host in "host_obj".
        """
        match = False
        all_hosts = list()
        try:
            rc, all_hosts = self.request('storage-systems/%s/hosts' % self.ssid)
        except Exception as err:
            self.module.fail_json(msg='Failed to determine host existence. Array Id [%s]. Error [%s].' % (self.ssid, to_native(err)))
        for host in all_hosts:
            for port in host['hostSidePorts']:
                port['type'] = port['type'].lower()
                port['address'] = port['address'].lower()
            ports = dict(((port['label'], port['id']) for port in host['ports']))
            ports.update(dict(((port['label'], port['id']) for port in host['initiators'])))
            for host_side_port in host['hostSidePorts']:
                if host_side_port['label'] in ports:
                    host_side_port['id'] = ports[host_side_port['label']]
            if host['label'].lower() == self.name.lower():
                self.host_obj = host
                match = True
        self.all_hosts = all_hosts
        return match

    @property
    def needs_update(self):
        """Determine whether we need to update the Host object
        As a side effect, we will set the ports that we need to update (portsForUpdate), and the ports we need to add
        (newPorts), on self.
        """
        changed = False
        if self.host_obj['hostTypeIndex'] != self.host_type_index:
            changed = True
        current_host_ports = dict(((port['id'], {'type': port['type'], 'port': port['address'], 'label': port['label']}) for port in self.host_obj['hostSidePorts']))
        if self.ports:
            for port in self.ports:
                for current_host_port_id in current_host_ports.keys():
                    if port == current_host_ports[current_host_port_id]:
                        current_host_ports.pop(current_host_port_id)
                        break
                    elif port['port'] == current_host_ports[current_host_port_id]['port']:
                        if self.port_on_diff_host(port) and (not self.force_port):
                            self.module.fail_json(msg='The port you specified [%s] is associated with a different host. Specify force_port as True or try a different port spec' % port)
                        if port['label'] != current_host_ports[current_host_port_id]['label'] or port['type'] != current_host_ports[current_host_port_id]['type']:
                            current_host_ports.pop(current_host_port_id)
                            self.ports_for_update.append({'portRef': current_host_port_id, 'port': port['port'], 'label': port['label'], 'hostRef': self.host_obj['hostRef']})
                            break
                else:
                    self.new_ports.append(port)
            self.ports_for_removal = list(current_host_ports.keys())
            changed = any([self.new_ports, self.ports_for_update, self.ports_for_removal, changed])
        return changed

    def port_on_diff_host(self, arg_port):
        """ Checks to see if a passed in port arg is present on a different host"""
        for host in self.all_hosts:
            if host['name'].lower() != self.name.lower():
                for port in host['hostSidePorts']:
                    if arg_port['label'].lower() == port['label'].lower() or arg_port['port'].lower() == port['address'].lower():
                        return True
        return False

    def update_host(self):
        self.post_body = {'name': self.name, 'hostType': {'index': self.host_type_index}}
        if self.ports:
            self.assigned_host_ports(apply_unassigning=True)
            self.post_body['portsToUpdate'] = self.ports_for_update
            self.post_body['portsToRemove'] = self.ports_for_removal
            self.post_body['ports'] = self.new_ports
        if not self.check_mode:
            try:
                rc, self.host_obj = self.request('storage-systems/%s/hosts/%s' % (self.ssid, self.host_obj['id']), method='POST', data=self.post_body, ignore_errors=True)
            except Exception as err:
                self.module.fail_json(msg='Failed to update host. Array Id [%s]. Error [%s].' % (self.ssid, to_native(err)))
        self.module.exit_json(changed=True)

    def create_host(self):
        self.assigned_host_ports(apply_unassigning=True)
        post_body = dict(name=self.name, hostType=dict(index=self.host_type_index))
        if self.ports:
            post_body.update(ports=self.ports)
        if not self.host_exists:
            if not self.check_mode:
                try:
                    rc, self.host_obj = self.request('storage-systems/%s/hosts' % self.ssid, method='POST', data=post_body, ignore_errors=True)
                except Exception as err:
                    self.module.fail_json(msg='Failed to create host. Array Id [%s]. Error [%s].' % (self.ssid, to_native(err)))
        else:
            payload = self.build_success_payload(self.host_obj)
            self.module.exit_json(changed=False, msg='Host already exists. Id [%s]. Host [%s].' % (self.ssid, self.name), **payload)
        payload = self.build_success_payload(self.host_obj)
        self.module.exit_json(changed=True, msg='Host created.')

    def remove_host(self):
        try:
            rc, resp = self.request('storage-systems/%s/hosts/%s' % (self.ssid, self.host_obj['id']), method='DELETE')
        except Exception as err:
            self.module.fail_json(msg='Failed to remove host.  Host[%s]. Array Id [%s]. Error [%s].' % (self.host_obj['id'], self.ssid, to_native(err)))

    def build_success_payload(self, host=None):
        keys = []
        if host:
            result = dict(((key, host[key]) for key in keys))
        else:
            result = dict()
        result['ssid'] = self.ssid
        result['api_url'] = self.url
        return result

    def apply(self):
        if self.state == 'present':
            if self.host_type_index is None:
                self.host_type_index = self.default_host_type
            self.check_port_types()
            if self.host_exists:
                if self.needs_update and self.valid_host_type:
                    self.update_host()
                else:
                    payload = self.build_success_payload(self.host_obj)
                    self.module.exit_json(changed=False, msg='Host already present; no changes required.', **payload)
            elif self.valid_host_type:
                self.create_host()
        else:
            payload = self.build_success_payload()
            if self.host_exists:
                self.remove_host()
                self.module.exit_json(changed=True, msg='Host removed.', **payload)
            else:
                self.module.exit_json(changed=False, msg='Host already absent.', **payload)