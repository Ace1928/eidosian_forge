from __future__ import absolute_import, division, print_function
import_profile:
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.compat.version import LooseVersion
from ansible_collections.community.vmware.plugins.module_utils.vmware_rest_client import VmwareRestClient
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi
import json
import time
class VcenterProfile(VmwareRestClient):

    def __init__(self, module):
        super(VcenterProfile, self).__init__(module)
        self.config_path = self.params['config_path']

    def list_vc_infraprofile_configs(self):
        profile_configs_list = self.api_client.appliance.infraprofile.Configs.list()
        config_list = []
        for x in profile_configs_list:
            config_list.append({'info': x.info, 'name': x.name})
        self.module.exit_json(changed=False, infra_configs_list=config_list)

    def get_profile_spec(self):
        infra = self.api_client.appliance.infraprofile.Configs
        profiles = {}
        profiles = self.params['profiles'].split(',')
        profile_spec = infra.ProfilesSpec(encryption_key='encryption_key', description='description', profiles=set(profiles))
        return profile_spec

    def vc_export_profile_task(self):
        profile_spec = self.get_profile_spec()
        infra = self.api_client.appliance.infraprofile.Configs
        config_json = infra.export(spec=profile_spec)
        if self.config_path is None:
            self.config_path = self.params.get('api') + '.json'
        parsed = json.loads(config_json)
        with open(self.config_path, 'w', encoding='utf-8') as outfile:
            json.dump(parsed, outfile, ensure_ascii=False, indent=2)
        self.module.exit_json(changed=False, export_config_json=config_json)

    def read_profile(self):
        with open(self.config_path, 'r') as file:
            return file.read()

    def get_import_profile_spec(self):
        infra = self.api_client.appliance.infraprofile.Configs
        config_spec = self.read_profile()
        profile_spec = self.get_profile_spec()
        import_profile_spec = infra.ImportProfileSpec(config_spec=config_spec, profile_spec=profile_spec)
        return import_profile_spec

    def vc_import_profile_task(self):
        infra = self.api_client.appliance.infraprofile.Configs
        import_profile_spec = self.get_import_profile_spec()
        import_task = infra.import_profile_task(import_profile_spec)
        self.wait_for_task(import_task)
        if 'SUCCEEDED' == import_task.get_info().status:
            self.module.exit_json(changed=True, status=import_task.get_info().result.value)
        self.module.fail_json(msg='Failed to import profile status:"%s" ' % import_task.get_info().status)

    def vc_validate_profile_task(self):
        infra = self.api_client.appliance.infraprofile.Configs
        import_profile_spec = self.get_import_profile_spec()
        validate_task = infra.validate_task(import_profile_spec)
        if 'VALID' == validate_task.get_info().result.get_field('status').value:
            self.module.exit_json(changed=False, status=validate_task.get_info().result.get_field('status').value)
        elif 'INVALID' == validate_task.get_info().result.get_field('status').value:
            self.module.exit_json(changed=False, status=validate_task.get_info().result.get_field('status').value)
        else:
            self.module.fail_json(msg='Failed to validate profile status:"%s" ' % dir(validate_task.get_info().status))

    def wait_for_task(self, task, poll_interval=1):
        while task.get_info().status == 'RUNNING':
            time.sleep(poll_interval)