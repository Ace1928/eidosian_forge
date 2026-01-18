from __future__ import absolute_import, division, print_function
import re
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
from ansible.module_utils._text import to_native
class NetAppESeriesIscsiInterface(NetAppESeriesModule):

    def __init__(self):
        ansible_options = dict(controller=dict(type='str', required=True, choices=['A', 'B']), port=dict(type='int', required=True), state=dict(type='str', required=False, default='enabled', choices=['enabled', 'disabled']), address=dict(type='str', required=False), subnet_mask=dict(type='str', required=False), gateway=dict(type='str', required=False), config_method=dict(type='str', required=False, default='dhcp', choices=['dhcp', 'static']), mtu=dict(type='int', default=1500, required=False, aliases=['max_frame_size']), speed=dict(type='str', required=False))
        required_if = [['config_method', 'static', ['address', 'subnet_mask']]]
        super(NetAppESeriesIscsiInterface, self).__init__(ansible_options=ansible_options, web_services_version='02.00.0000.0000', required_if=required_if, supports_check_mode=True)
        args = self.module.params
        self.controller = args['controller']
        self.port = args['port']
        self.mtu = args['mtu']
        self.state = args['state']
        self.address = args['address']
        self.subnet_mask = args['subnet_mask']
        self.gateway = args['gateway']
        self.config_method = args['config_method']
        self.speed = args['speed']
        self.check_mode = self.module.check_mode
        self.post_body = dict()
        self.controllers = list()
        self.get_target_interface_cache = None
        if self.mtu < 1500 or self.mtu > 9000:
            self.module.fail_json(msg='The provided mtu is invalid, it must be > 1500 and < 9000 bytes.')
        if self.config_method == 'dhcp' and any([self.address, self.subnet_mask, self.gateway]):
            self.module.fail_json(msg='A config_method of dhcp is mutually exclusive with the address, subnet_mask, and gateway options.')
        address_regex = re.compile('^\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}$')
        if self.address and (not address_regex.match(self.address)):
            self.module.fail_json(msg='An invalid ip address was provided for address.')
        if self.subnet_mask and (not address_regex.match(self.subnet_mask)):
            self.module.fail_json(msg='An invalid ip address was provided for subnet_mask.')
        if self.gateway and (not address_regex.match(self.gateway)):
            self.module.fail_json(msg='An invalid ip address was provided for gateway.')
        self.get_host_board_id_cache = None

    @property
    def interfaces(self):
        ifaces = list()
        try:
            rc, ifaces = self.request('storage-systems/%s/graph/xpath-filter?query=/controller/hostInterfaces' % self.ssid)
        except Exception as err:
            self.module.fail_json(msg='Failed to retrieve defined host interfaces. Array Id [%s]. Error [%s].' % (self.ssid, to_native(err)))
        iscsi_interfaces = []
        for iface in [iface for iface in ifaces if iface['interfaceType'] == 'iscsi']:
            if iface['iscsi']['interfaceData']['type'] == 'ethernet':
                iscsi_interfaces.append(iface)
        return iscsi_interfaces

    def get_host_board_id(self, iface_ref):
        if self.get_host_board_id_cache is None:
            try:
                rc, iface_board_map_list = self.request('storage-systems/%s/graph/xpath-filter?query=/ioInterfaceHicMap' % self.ssid)
            except Exception as err:
                self.module.fail_json(msg='Failed to retrieve IO interface HIC mappings! Array Id [%s]. Error [%s].' % (self.ssid, to_native(err)))
            self.get_host_board_id_cache = dict()
            for iface_board_map in iface_board_map_list:
                self.get_host_board_id_cache.update({iface_board_map['interfaceRef']: iface_board_map['hostBoardRef']})
        return self.get_host_board_id_cache[iface_ref]

    def get_controllers(self):
        """Retrieve a mapping of controller labels to their references
        {
            "A": "070000000000000000000001",
            "B": "070000000000000000000002",
        }
        :return: the controllers defined on the system
        """
        controllers = list()
        try:
            rc, controllers = self.request('storage-systems/%s/graph/xpath-filter?query=/controller/id' % self.ssid)
        except Exception as err:
            self.module.fail_json(msg='Failed to retrieve controller list! Array Id [%s]. Error [%s].' % (self.ssid, to_native(err)))
        controllers.sort()
        controllers_dict = {}
        i = ord('A')
        for controller in controllers:
            label = chr(i)
            controllers_dict[label] = controller
            i += 1
        return controllers_dict

    def get_target_interface(self):
        """Retrieve the specific controller iSCSI interface."""
        if self.get_target_interface_cache is None:
            ifaces = self.interfaces
            controller_ifaces = []
            for iface in ifaces:
                if self.controllers[self.controller] == iface['iscsi']['controllerId']:
                    controller_ifaces.append([iface['iscsi']['channel'], iface, iface['iscsi']['interfaceData']['ethernetData']['linkStatus']])
            sorted_controller_ifaces = sorted(controller_ifaces)
            if self.port < 1 or self.port > len(controller_ifaces):
                status_msg = ', '.join(['%s (link %s)' % (index + 1, values[2]) for index, values in enumerate(sorted_controller_ifaces)])
                self.module.fail_json(msg='Invalid controller %s iSCSI port. Available ports: %s, Array Id [%s].' % (self.controller, status_msg, self.ssid))
            self.get_target_interface_cache = sorted_controller_ifaces[self.port - 1][1]
        return self.get_target_interface_cache

    def make_update_body(self, target_iface):
        target_iface = target_iface['iscsi']
        body = dict(iscsiInterface=target_iface['id'])
        update_required = False
        if self.state == 'enabled':
            settings = dict()
            if not target_iface['ipv4Enabled']:
                update_required = True
                settings['ipv4Enabled'] = [True]
            if self.mtu != target_iface['interfaceData']['ethernetData']['maximumFramePayloadSize']:
                update_required = True
                settings['maximumFramePayloadSize'] = [self.mtu]
            if self.config_method == 'static':
                ipv4Data = target_iface['ipv4Data']['ipv4AddressData']
                if ipv4Data['ipv4Address'] != self.address:
                    update_required = True
                    settings['ipv4Address'] = [self.address]
                if ipv4Data['ipv4SubnetMask'] != self.subnet_mask:
                    update_required = True
                    settings['ipv4SubnetMask'] = [self.subnet_mask]
                if self.gateway is not None and ipv4Data['ipv4GatewayAddress'] != self.gateway:
                    update_required = True
                    settings['ipv4GatewayAddress'] = [self.gateway]
                if target_iface['ipv4Data']['ipv4AddressConfigMethod'] != 'configStatic':
                    update_required = True
                    settings['ipv4AddressConfigMethod'] = ['configStatic']
            elif target_iface['ipv4Data']['ipv4AddressConfigMethod'] != 'configDhcp':
                update_required = True
                settings.update(dict(ipv4Enabled=[True], ipv4AddressConfigMethod=['configDhcp']))
            body['settings'] = settings
        elif target_iface['ipv4Enabled']:
            update_required = True
            body['settings'] = dict(ipv4Enabled=[False])
        return (update_required, body)

    def make_update_speed_body(self, target_iface):
        target_iface = target_iface['iscsi']
        if self.speed is None:
            return (False, dict())
        else:
            if target_iface['interfaceData']['ethernetData']['autoconfigSupport']:
                self.module.warn("This interface's HIC speed is autoconfigured!")
                return (False, dict())
            if self.speed == strip_interface_speed(target_iface['interfaceData']['ethernetData']['currentInterfaceSpeed']):
                return (False, dict())
        supported_speeds = dict()
        for supported_speed in target_iface['interfaceData']['ethernetData']['supportedInterfaceSpeeds']:
            supported_speeds.update({strip_interface_speed(supported_speed): supported_speed})
        if self.speed not in supported_speeds:
            self.module.fail_json(msg='The host interface card (HIC) does not support the provided speed. Array Id [%s]. Supported speeds [%s]' % (self.ssid, ', '.join(supported_speeds.keys())))
        body = {'settings': {'maximumInterfaceSpeed': [supported_speeds[self.speed]]}, 'portsRef': {}}
        hic_ref = self.get_host_board_id(target_iface['id'])
        if hic_ref == '0000000000000000000000000000000000000000':
            body.update({'portsRef': {'portRefType': 'baseBoard', 'baseBoardRef': target_iface['id'], 'hicRef': ''}})
        else:
            body.update({'portsRef': {'portRefType': 'hic', 'hicRef': hic_ref, 'baseBoardRef': ''}})
        return (True, body)

    def update(self):
        self.controllers = self.get_controllers()
        if self.controller not in self.controllers:
            self.module.fail_json(msg='The provided controller name is invalid. Valid controllers: %s.' % ', '.join(self.controllers.keys()))
        iface_before = self.get_target_interface()
        update_required, body = self.make_update_body(iface_before)
        if update_required and (not self.check_mode):
            try:
                rc, result = self.request('storage-systems/%s/symbol/setIscsiInterfaceProperties' % self.ssid, method='POST', data=body, ignore_errors=True)
                if rc == 422 and result['retcode'] in ['busy', '3']:
                    self.module.fail_json(msg='The interface is currently busy (probably processing a previously requested modification request). This operation cannot currently be completed. Array Id [%s]. Error [%s].' % (self.ssid, result))
                elif rc != 200:
                    self.module.fail_json(msg='Failed to modify the interface! Array Id [%s]. Error [%s].' % (self.ssid, to_native(result)))
            except Exception as err:
                self.module.fail_json(msg='Connection failure: we failed to modify the interface! Array Id [%s]. Error [%s].' % (self.ssid, to_native(err)))
        update_speed_required, speed_body = self.make_update_speed_body(iface_before)
        if update_speed_required and (not self.check_mode):
            try:
                rc, result = self.request('storage-systems/%s/symbol/setHostPortsAttributes?verboseErrorResponse=true' % self.ssid, method='POST', data=speed_body)
            except Exception as err:
                self.module.fail_json(msg='Failed to update host interface card speed. Array Id [%s], Body [%s]. Error [%s].' % (self.ssid, speed_body, to_native(err)))
        if update_required or update_speed_required:
            self.module.exit_json(msg='The interface settings have been updated.', changed=True)
        self.module.exit_json(msg='No changes were required.', changed=False)