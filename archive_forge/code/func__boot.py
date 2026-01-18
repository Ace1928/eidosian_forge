import base64
import collections
from urllib import parse
from novaclient import api_versions
from novaclient import base
from novaclient import crypto
from novaclient import exceptions
from novaclient.i18n import _
def _boot(self, response_key, name, image, flavor, meta=None, files=None, userdata=None, reservation_id=False, return_raw=False, min_count=None, max_count=None, security_groups=None, key_name=None, availability_zone=None, block_device_mapping=None, block_device_mapping_v2=None, nics=None, scheduler_hints=None, config_drive=None, admin_pass=None, disk_config=None, access_ip_v4=None, access_ip_v6=None, description=None, tags=None, trusted_image_certificates=None, host=None, hypervisor_hostname=None, hostname=None):
    """
        Create (boot) a new server.
        """
    body = {'server': {'name': name, 'imageRef': str(base.getid(image)) if image else '', 'flavorRef': str(base.getid(flavor))}}
    if userdata:
        body['server']['user_data'] = self.transform_userdata(userdata)
    if meta:
        body['server']['metadata'] = meta
    if reservation_id:
        body['server']['return_reservation_id'] = reservation_id
        return_raw = True
    if key_name:
        body['server']['key_name'] = key_name
    if scheduler_hints:
        body['os:scheduler_hints'] = scheduler_hints
    if config_drive:
        body['server']['config_drive'] = config_drive
    if admin_pass:
        body['server']['adminPass'] = admin_pass
    if not min_count:
        min_count = 1
    if not max_count:
        max_count = min_count
    body['server']['min_count'] = min_count
    body['server']['max_count'] = max_count
    if security_groups:
        body['server']['security_groups'] = [{'name': sg} for sg in security_groups]
    if files:
        personality = body['server']['personality'] = []
        for filepath, file_or_string in sorted(files.items(), key=lambda x: x[0]):
            if hasattr(file_or_string, 'read'):
                data = file_or_string.read()
            else:
                data = file_or_string
            if isinstance(data, str):
                data = data.encode('utf-8')
            cont = base64.b64encode(data).decode('utf-8')
            personality.append({'path': filepath, 'contents': cont})
    if availability_zone:
        body['server']['availability_zone'] = availability_zone
    if block_device_mapping:
        body['server']['block_device_mapping'] = self._parse_block_device_mapping(block_device_mapping)
    elif block_device_mapping_v2:
        if image:
            bdm_dict = {'uuid': base.getid(image), 'source_type': 'image', 'destination_type': 'local', 'boot_index': 0, 'delete_on_termination': True}
            block_device_mapping_v2.insert(0, bdm_dict)
        body['server']['block_device_mapping_v2'] = block_device_mapping_v2
    if nics is not None:
        if isinstance(nics, str):
            all_net_data = nics
        else:
            all_net_data = []
            for nic_info in nics:
                net_data = {}
                if nic_info.get('net-id'):
                    net_data['uuid'] = nic_info['net-id']
                if nic_info.get('v4-fixed-ip') and nic_info.get('v6-fixed-ip'):
                    raise base.exceptions.CommandError(_("Only one of 'v4-fixed-ip' and 'v6-fixed-ip' may be provided."))
                elif nic_info.get('v4-fixed-ip'):
                    net_data['fixed_ip'] = nic_info['v4-fixed-ip']
                elif nic_info.get('v6-fixed-ip'):
                    net_data['fixed_ip'] = nic_info['v6-fixed-ip']
                if nic_info.get('port-id'):
                    net_data['port'] = nic_info['port-id']
                if nic_info.get('tag'):
                    net_data['tag'] = nic_info['tag']
                all_net_data.append(net_data)
        body['server']['networks'] = all_net_data
    if disk_config is not None:
        body['server']['OS-DCF:diskConfig'] = disk_config
    if access_ip_v4 is not None:
        body['server']['accessIPv4'] = access_ip_v4
    if access_ip_v6 is not None:
        body['server']['accessIPv6'] = access_ip_v6
    if description:
        body['server']['description'] = description
    if tags:
        body['server']['tags'] = tags
    if trusted_image_certificates:
        body['server']['trusted_image_certificates'] = trusted_image_certificates
    if host:
        body['server']['host'] = host
    if hypervisor_hostname:
        body['server']['hypervisor_hostname'] = hypervisor_hostname
    if hostname:
        body['server']['hostname'] = hostname
    return self._create('/servers', body, response_key, return_raw=return_raw)