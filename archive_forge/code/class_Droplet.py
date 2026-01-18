from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule, env_fallback
class Droplet(JsonfyMixIn):
    manager = None

    def __init__(self, droplet_json):
        self.status = 'new'
        self.__dict__.update(droplet_json)

    def is_powered_on(self):
        return self.status == 'active'

    def update_attr(self, attrs=None):
        if attrs:
            for k, v in attrs.items():
                setattr(self, k, v)
            networks = attrs.get('networks', {})
            for network in networks.get('v6', []):
                if network['type'] == 'public':
                    setattr(self, 'public_ipv6_address', network['ip_address'])
                else:
                    setattr(self, 'private_ipv6_address', network['ip_address'])
        else:
            json = self.manager.show_droplet(self.id)
            if json['ip_address']:
                self.update_attr(json)

    def power_on(self):
        if self.status != 'off':
            raise AssertionError('Can only power on a closed one.')
        json = self.manager.power_on_droplet(self.id)
        self.update_attr(json)

    def ensure_powered_on(self, wait=True, wait_timeout=300):
        if self.is_powered_on():
            return
        if self.status == 'off':
            self.power_on()
        if wait:
            end_time = time.monotonic() + wait_timeout
            while time.monotonic() < end_time:
                time.sleep(10)
                self.update_attr()
                if self.is_powered_on():
                    if not self.ip_address:
                        raise TimeoutError('No ip is found.', self.id)
                    return
            raise TimeoutError('Wait for droplet running timeout', self.id)

    def destroy(self):
        return self.manager.destroy_droplet(self.id, scrub_data=True)

    @classmethod
    def setup(cls, api_token):
        cls.manager = DoManager(None, api_token, api_version=2)

    @classmethod
    def add(cls, name, size_id, image_id, region_id, ssh_key_ids=None, virtio=True, private_networking=False, backups_enabled=False, user_data=None, ipv6=False):
        private_networking_lower = str(private_networking).lower()
        backups_enabled_lower = str(backups_enabled).lower()
        ipv6_lower = str(ipv6).lower()
        json = cls.manager.new_droplet(name, size_id, image_id, region_id, ssh_key_ids=ssh_key_ids, virtio=virtio, private_networking=private_networking_lower, backups_enabled=backups_enabled_lower, user_data=user_data, ipv6=ipv6_lower)
        droplet = cls(json)
        return droplet

    @classmethod
    def find(cls, id=None, name=None):
        if not id and (not name):
            return False
        droplets = cls.list_all()
        for droplet in droplets:
            if droplet.id == id:
                return droplet
        for droplet in droplets:
            if droplet.name == name:
                return droplet
        return False

    @classmethod
    def list_all(cls):
        json = cls.manager.all_active_droplets()
        return list(map(cls, json))