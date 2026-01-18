from __future__ import absolute_import, division, print_function
import re
import uuid
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves import xrange
from ansible.module_utils.common.text.converters import to_native
def _create_machine(module, profitbricks, datacenter, name):
    cores = module.params.get('cores')
    ram = module.params.get('ram')
    cpu_family = module.params.get('cpu_family')
    volume_size = module.params.get('volume_size')
    disk_type = module.params.get('disk_type')
    image_password = module.params.get('image_password')
    ssh_keys = module.params.get('ssh_keys')
    bus = module.params.get('bus')
    lan = module.params.get('lan')
    assign_public_ip = module.params.get('assign_public_ip')
    subscription_user = module.params.get('subscription_user')
    subscription_password = module.params.get('subscription_password')
    location = module.params.get('location')
    image = module.params.get('image')
    assign_public_ip = module.boolean(module.params.get('assign_public_ip'))
    wait = module.params.get('wait')
    wait_timeout = module.params.get('wait_timeout')
    if assign_public_ip:
        public_found = False
        lans = profitbricks.list_lans(datacenter)
        for lan in lans['items']:
            if lan['properties']['public']:
                public_found = True
                lan = lan['id']
        if not public_found:
            i = LAN(name='public', public=True)
            lan_response = profitbricks.create_lan(datacenter, i)
            _wait_for_completion(profitbricks, lan_response, wait_timeout, '_create_machine')
            lan = lan_response['id']
    v = Volume(name=str(uuid.uuid4()).replace('-', '')[:10], size=volume_size, image=image, image_password=image_password, ssh_keys=ssh_keys, disk_type=disk_type, bus=bus)
    n = NIC(lan=int(lan))
    s = Server(name=name, ram=ram, cores=cores, cpu_family=cpu_family, create_volumes=[v], nics=[n])
    try:
        create_server_response = profitbricks.create_server(datacenter_id=datacenter, server=s)
        _wait_for_completion(profitbricks, create_server_response, wait_timeout, 'create_virtual_machine')
        server_response = profitbricks.get_server(datacenter_id=datacenter, server_id=create_server_response['id'], depth=3)
    except Exception as e:
        module.fail_json(msg='failed to create the new server: %s' % str(e))
    else:
        return server_response