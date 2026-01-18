from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.elementsw.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.elementsw.plugins.module_utils.netapp_elementsw_module import NaElementSWModule
import time
class ElementSWBackup(object):
    """ class to handle backup operations """

    def __init__(self):
        """
            Setup Ansible parameters and SolidFire connection
        """
        self.argument_spec = netapp_utils.ontap_sf_host_argument_spec()
        self.argument_spec.update(dict(src_volume_id=dict(aliases=['volume_id'], required=True, type='str'), dest_hostname=dict(required=False, type='str'), dest_username=dict(required=False, type='str'), dest_password=dict(required=False, type='str', no_log=True), dest_volume_id=dict(required=True, type='str'), format=dict(required=False, choices=['native', 'uncompressed'], default='native'), script=dict(required=False, type='str'), script_parameters=dict(required=False, type='dict')))
        self.module = AnsibleModule(argument_spec=self.argument_spec, required_together=[['script', 'script_parameters']], supports_check_mode=True)
        if HAS_SF_SDK is False:
            self.module.fail_json(msg='Unable to import the SolidFire Python SDK')
        if self.module.params['dest_hostname'] is None:
            self.module.params['dest_hostname'] = self.module.params['hostname']
        if self.module.params['dest_username'] is None:
            self.module.params['dest_username'] = self.module.params['username']
        if self.module.params['dest_password'] is None:
            self.module.params['dest_password'] = self.module.params['password']
        params = self.module.params
        self.src_connection = netapp_utils.create_sf_connection(self.module)
        self.module.params['username'] = params['dest_username']
        self.module.params['password'] = params['dest_password']
        self.module.params['hostname'] = params['dest_hostname']
        self.dest_connection = netapp_utils.create_sf_connection(self.module)
        self.elementsw_helper = NaElementSWModule(self.src_connection)
        self.attributes = self.elementsw_helper.set_element_attributes(source='na_elementsw_backup')

    def apply(self):
        """
            Apply backup creation logic
        """
        self.create_backup()
        self.module.exit_json(changed=True)

    def create_backup(self):
        """
            Create backup
        """
        try:
            write_obj = self.dest_connection.start_bulk_volume_write(volume_id=self.module.params['dest_volume_id'], format=self.module.params['format'], attributes=self.attributes)
            write_key = write_obj.key
        except solidfire.common.ApiServerError as err:
            self.module.fail_json(msg='Error starting bulk write on destination cluster', exception=to_native(err))
        if self.module.params['script'] is None and self.module.params['script_parameters'] is None:
            self.module.params['script'] = 'bv_internal.py'
            self.module.params['script_parameters'] = {'write': {'mvip': self.module.params['dest_hostname'], 'username': self.module.params['dest_username'], 'password': self.module.params['dest_password'], 'key': write_key, 'endpoint': 'solidfire', 'format': self.module.params['format']}, 'range': {'lba': 0, 'blocks': 244224}}
        try:
            read_obj = self.src_connection.start_bulk_volume_read(self.module.params['src_volume_id'], self.module.params['format'], script=self.module.params['script'], script_parameters=self.module.params['script_parameters'], attributes=self.attributes)
        except solidfire.common.ApiServerError as err:
            self.module.fail_json(msg='Error starting bulk read on source cluster', exception=to_native(err))
        completed = False
        while completed is not True:
            time.sleep(2)
            try:
                result = self.src_connection.get_async_result(read_obj.async_handle, True)
            except solidfire.common.ApiServerError as err:
                self.module.fail_json(msg='Unable to check backup job status', exception=to_native(err))
            if result['status'] != 'running':
                completed = True
        if 'error' in result:
            self.module.fail_json(msg=result['error']['message'])